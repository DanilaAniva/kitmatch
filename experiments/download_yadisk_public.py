#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

API_BASE = "https://cloud-api.yandex.net/v1/disk/public"

def http_get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "yd-public-downloader"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get_stream(url):
    req = urllib.request.Request(url, headers={"User-Agent": "yd-public-downloader"})
    return urllib.request.urlopen(req)

def build_url(endpoint, params):
    return f"{API_BASE}{endpoint}?{urllib.parse.urlencode(params)}"

def get_meta(public_url, path=None, limit=None, offset=None):
    params = {"public_key": public_url}
    if path is not None:
        params["path"] = path
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    url = build_url("/resources", params)
    return http_get_json(url)

def get_download_href(public_url, path=None):
    params = {"public_key": public_url}
    if path is not None:
        params["path"] = path
    url = build_url("/resources/download", params)
    data = http_get_json(url)
    return data["href"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def download_file(public_url, path, dest_path):
    href = get_download_href(public_url, path)
    ensure_dir(os.path.dirname(dest_path) or ".")
    with http_get_stream(href) as r, open(dest_path, "wb") as f:
        total = int(r.headers.get("Content-Length", "0"))
        downloaded = 0
        chunk = 1024 * 256
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                done = downloaded * 100 // total
                sys.stderr.write(f"\r[{done:3d}%] {dest_path}")
                sys.stderr.flush()
        sys.stderr.write(f"\r[100%] {dest_path}\n")

def iter_dir(public_url, path="/"):
    # Корректная пагинация: забираем все страницы каталога
    limit = 1000  # максимум, который понимает API
    offset = 0
    while True:
        meta = get_meta(public_url, path=path, limit=limit, offset=offset)
        embedded = meta.get("_embedded", {})
        items = embedded.get("items", [])
        if not items:
            break
        for it in items:
            yield it
        if len(items) < limit:
            break
        offset += limit

def walk_and_download(public_url, dest_dir):
    root = get_meta(public_url)
    rtype = root.get("type")
    root_name = root.get("name", "download")

    if rtype == "file":
        local = os.path.join(dest_dir, root_name)
        print(f"Downloading file: {root_name}")
        download_file(public_url, None, local)
        return

    if rtype != "dir":
        raise RuntimeError(f"Неизвестный тип ресурса: {rtype}")

    # Для каталога: обходим рекурсивно
    print(f"Listing folder: {root_name}")
    stack = ["/"]
    while stack:
        current = stack.pop()
        for item in iter_dir(public_url, path=current):
            itype = item.get("type")
            ipath = item.get("path")  # относительный путь внутри публичного ресурса (начинается с "/")
            name = item.get("name")

            if itype == "dir":
                stack.append(ipath)
            elif itype == "file":
                rel = ipath.lstrip("/")
                local = os.path.join(dest_dir, rel)
                print(f"Downloading: {rel}")
                download_file(public_url, ipath, local)
            else:
                print(f"Пропуск: {itype} {ipath}")

def main():
    ap = argparse.ArgumentParser(description="Скачивание публичного ресурса Яндекс.Диска (файл или папка)")
    ap.add_argument("public_url", help="Публичная ссылка, например: https://disk.yandex.ru/d/ZnYSBHK9gGLTtQ")
    ap.add_argument("-o", "--out", default="dataset", help="Папка назначения (по умолчанию: dataset)")
    args = ap.parse_args()

    ensure_dir(args.out)
    try:
        walk_and_download(args.public_url, args.out)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        print(f"HTTPError {e.code}: {msg}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()