import React, { useState, useRef } from "react";
import { Link } from "react-router-dom";

export default function BatchPage() {
  const [zipFile, setZipFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null); // Изменено на null
  const [threshold, setThreshold] = useState(0.5); // Состояние для порога
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    if (!e.target.files || !e.target.files[0]) return;
    handleFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleFile = (file) => {
    if (file.type !== 'application/zip' && !file.name.endsWith('.zip')) {
      setError('Пожалуйста, выберите ZIP файл');
      return;
    }
    setZipFile(file);
    setError(null);
    processArchive(file);
  };

  const processArchive = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null); // Сбрасываем предыдущие результаты
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence_threshold', threshold); // Отправляем порог

    try {
      const res = await fetch('/api/v1/infer/archive', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errorMessage = `Ошибка сервера: ${res.status}`;
        try {
          const errorData = await res.json();
          errorMessage = errorData.error || errorData.detail || errorMessage;
        } catch (e) {}
        throw new Error(errorMessage);
      }
      
      const data = await res.json();
      setResults(data);

    } catch (err) {
      setError('Ошибка при обработке архива: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="batch-app">
      <div className="batch-card">
        <div className="batch-upload-section">
          <h2 className="batch-title"> Обработка архива изображений</h2>
          <p className="batch-subtitle">
            Загрузите ZIP архив с изображениями для массовой обработки
          </p>
          
          {/* Слайдер для порога */}
          <div className="processing-threshold-slider" style={{margin: '1rem 0'}}>
            <label htmlFor="threshold">Порог уверенности: {Math.round(threshold * 100)}%</label>
            <input
              type="range"
              id="threshold"
              min="0.01"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              disabled={loading}
              style={{width: '100%'}}
            />
          </div>

          <div className="batch-dropArea" onDrop={handleDrop} onDragOver={handleDragOver}>
            {zipFile ? (
              <div className="batch-file-info">
                <div className="batch-file-icon"></div>
                <div className="batch-file-details">
                  <div className="batch-file-name">{zipFile.name}</div>
                  <div className="batch-file-size">
                    {(zipFile.size / (1024 * 1024)).toFixed(2)} MB
                  </div>
                </div>
              </div>
            ) : (
              <div className="batch-placeholder">
                <p> Перетащите ZIP архив сюда</p>
                <p>или</p>
                <p>Кликните, чтобы выбрать файл</p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              className="batch-fileInput"
              onChange={handleFileSelect}
            />
          </div>

          {error && <div className="batch-error"> {error}</div>}
        </div>

        <div className="batch-results-section">
          <h3> Результаты обработки</h3>
          <div className="batch-results-box">
            {loading ? (
              <div style={{textAlign: 'center', color: '#F85A40'}}>
                 Обработка архива... (Это может занять несколько минут)
              </div>
            ) : results ? (
              <>
                <div className="batch-status">
                  <p>Статус: {results.status === 'complete' ? 'Завершено' : 'Истекло время ожидания'}</p>
                  {results.message && <p>{results.message}</p>}
                </div>
                <ul className="batch-results-list">
                  {results.results && results.results.map((result, idx) => (
                    <li key={result.image_id || idx} className="batch-result-item">
                      <span className="filename">{result.original_filename || `image_${idx+1}`}</span>
                      <span className="tool-count">{result.bboxes ? result.bboxes.length : 0} объектов</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <div className="batch-empty">Загрузите архив для обработки</div>
            )}
          </div>

          <div className="batch-buttons">
            <button 
              className="batch-button"
              onClick={() => {
                setZipFile(null);
                setResults(null);
                setError(null);
                // Сбрасываем значение файлового инпута
                if (fileInputRef.current) {
                  fileInputRef.current.value = "";
                }
              }}
            >
               Очистить
            </button>
            <button 
              className="batch-button"
              disabled={!results || !results.results || results.results.length === 0}
              onClick={() => {
                const jsonString = JSON.stringify(results, null, 2);
                const blob = new Blob([jsonString], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `results_${zipFile ? zipFile.name.replace('.zip', '') : 'archive'}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
               Скачать отчёт
            </button>
          </div>

          <div style={{ marginTop: "20px", textAlign: "center" }}>
            <Link to="/">
              <button className="batch-button">⬅ Назад в меню</button>
            </Link>
          </div>
        </div>
      </div>

      {/* Логотип ЛЦТ */}
      <div className="lct-logo-bottom">
        <img 
          src="/assets/logo_lct.png" 
          alt="ЛЦТ" 
          className="lct-logo"
        />
      </div>
    </div>
  );
}