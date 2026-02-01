#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import random

API_BASE = "https://cloud-api.yandex.net/v1/disk/public"

def build_url(endpoint, params):
    return f"{API_BASE}{endpoint}?{urllib.parse.urlencode(params)}"

def http_get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "yd-public-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get_stream(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "yd-public-downloader"})
    return urllib.request.urlopen(req, timeout=timeout)

def get_meta(public_url, path=None, limit=None, offset=None, timeout=30):
    params = {"public_key": public_url}
    if path is not None:
        params["path"] = path
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    url = build_url("/resources", params)
    return http_get_json(url, timeout=timeout)

def get_download_href(public_url, path=None, timeout=30):
    params = {"public_key": public_url}
    if path is not None:
        params["path"] = path
    url = build_url("/resources/download", params)
    data = http_get_json(url, timeout=timeout)
    return data["href"]

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def list_dir_all_items(public_url, path="/", page_limit=1000, timeout=30):
    offset = 0
    while True:
        meta = get_meta(public_url, path=path, limit=page_limit, offset=offset, timeout=timeout)
        embedded = meta.get("_embedded", {})
        items = embedded.get("items", [])
        if not items:
            break
        for it in items:
            yield it
        if len(items) < page_limit:
            break
        offset += page_limit

def collect_all_files(public_url, timeout=30):
    root = get_meta(public_url, timeout=timeout)
    rtype = root.get("type")
    root_name = root.get("name", "download")

    if rtype == "file":
        size = root.get("size")
        return {
            "mode": "file",
            "root_name": root_name,
            "files": [{"path": None, "rel": root_name, "size": size}],
        }

    if rtype != "dir":
        raise RuntimeError(f"Неизвестный тип ресурса: {rtype}")

    files = []
    stack = ["/"]
    while stack:
        current = stack.pop()
        for item in list_dir_all_items(public_url, path=current, timeout=timeout):
            itype = item.get("type")
            ipath = item.get("path")
            if itype == "dir":
                stack.append(ipath)
            elif itype == "file":
                rel = (ipath or "").lstrip("/")
                files.append({"path": ipath, "rel": rel, "size": item.get("size")})
            else:
                pass

    return {"mode": "dir", "root_name": root_name, "files": files}

def human_size(n):
    if n is None:
        return "?"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024
        i += 1
    return f"{f:.1f}{units[i]}"

def download_one(public_url, item, out_dir, timeout, retries, backoff_base, skip_existing):
    ipath = item["path"]  # может быть None для одиночного файла-ссылки
    rel = item["rel"]
    size = item.get("size")

    dest = os.path.join(out_dir, rel)
    ensure_dir(os.path.dirname(dest))

    # Пропустить уже скачанный (по размеру)
    if skip_existing and os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return ("SKIP", rel, size, None)

    tmp = dest + ".part"
    attempt = 0
    while True:
        attempt += 1
        try:
            href = get_download_href(public_url, path=ipath, timeout=timeout)
            with http_get_stream(href, timeout=max(timeout, 60)) as r, open(tmp, "wb") as f:
                chunk = 1024 * 256
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
            os.replace(tmp, dest)
            return ("OK", rel, size, None)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt <= retries:
                delay = backoff_base * (2 ** (attempt - 1)) * (1 + 0.2 * random())
                time.sleep(delay)
                continue
            return ("ERR", rel, size, f"HTTP {e.code}")
        except Exception as e:
            if attempt <= retries:
                delay = backoff_base * (2 ** (attempt - 1)) * (1 + 0.2 * random())
                time.sleep(delay)
                continue
            return ("ERR", rel, size, str(e))
        finally:
            try:
                if os.path.exists(tmp) and (not os.path.exists(dest) or os.path.getsize(tmp) != os.path.getsize(dest)):
                    os.remove(tmp)
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser(description="Параллельное скачивание публичного ресурса Яндекс.Диска (файл или папка)")
    ap.add_argument("public_url", help="Публичная ссылка, например: https://disk.yandex.ru/d/ZnYSBHK9gGLTtQ")
    ap.add_argument("-o", "--out", default="dataset", help="Папка назначения (по умолчанию: dataset)")
    ap.add_argument("-w", "--workers", type=int, default=6, help="Количество параллельных потоков (по умолчанию: 6)")
    ap.add_argument("--timeout", type=int, default=30, help="Таймаут запросов, сек (по умолчанию: 30)")
    ap.add_argument("--retries", type=int, default=3, help="Количество повторов при сбоях (по умолчанию: 3)")
    ap.add_argument("--backoff", type=float, default=1.0, help="Начальная задержка бэкоффа, сек (по умолчанию: 1.0)")
    ap.add_argument("--no-skip-existing", action="store_true", help="Не пропускать уже скачанные файлы")
    args = ap.parse_args()

    skip_existing = not args.no_skip_existing
    os.makedirs(args.out, exist_ok=True)

    try:
        plan = collect_all_files(args.public_url, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        print(f"HTTPError {e.code}: {msg}", file=sys.stderr)
        sys.exit(1)

    files = plan["files"]
    if plan["mode"] == "file":
        print(f"Режим: одиночный файл: {files[0]['rel']} ({human_size(files[0].get('size'))})")
    else:
        total_sz = sum((f.get("size") or 0) for f in files)
        print(f"Режим: папка. Файлов: {len(files)}, суммарно: {human_size(total_sz)}")

    if not files:
        print("Нечего скачивать.")
        return

    started = 0
    ok = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for it in files:
            futures.append(ex.submit(
                download_one,
                args.public_url, it, args.out,
                args.timeout, args.retries, args.backoff, skip_existing
            ))
            started += 1

        for fut in as_completed(futures):
            status, rel, size, err = fut.result()
            if status == "OK":
                ok += 1
                print(f"[OK]   {rel} ({human_size(size)})")
            elif status == "SKIP":
                skipped += 1
                print(f"[SKIP] {rel} ({human_size(size)})")
            else:
                failed += 1
                print(f"[ERR]  {rel} ({human_size(size)}): {err}")

    print(f"Готово: ok={ok}, skip={skipped}, err={failed}, всего={started}")

if __name__ == "__main__":
    main()