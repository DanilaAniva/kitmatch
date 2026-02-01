import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// URL теперь относительный, Nginx будет проксировать запросы
const API_PREFIX = '/api/v1/infer';

// Дефолтный список инструментов
const DEFAULT_TOOLS = [
  'Отвертка «-»',
  'Отвертка «+»', 
  'Отвертка на смещенный крест',
  'Коловорот',
  'Пассатижи контровочные',
  'Пассатижи',
  'Шэрница',
  'Разводной ключ',
  'Открывашка для банок с маслом',
  'Ключ рожковый/накидной ¾',
  'Бокорезы'
];

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.5); // Состояние для порога
  const [filteredDetections, setFilteredDetections] = useState([]); // Состояние для отфильтрованных результатов
  const [isEditing, setIsEditing] = useState(false);
  const [editCounts, setEditCounts] = useState({});
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Убираем начальную загрузку моков, так как эндпоинт /items удален
    // Cleanup function для previewUrl
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, []); // Убираем previewUrl из зависимостей

  // Эффект для динамической фильтрации при изменении порога
  useEffect(() => {
    const filtered = detections.filter(d => d.confidence >= threshold);
    setFilteredDetections(filtered);
  }, [detections, threshold]);

  // Функция для преобразования файла в base64 data URI
  const fileToDataUri = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(file);
    });
  };

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
    // Проверяем тип файла
    if (!file.type.startsWith('image/')) {
      setError('Пожалуйста, выберите файл изображения');
      return;
    }

    // Проверяем размер файла (например, максимум 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setError('Файл слишком большой. Максимальный размер: 10MB');
      return;
    }

    setImageFile(file);
    setError(null);
    
    // Очищаем предыдущий preview URL
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    analyzeImage(file);
  };

  const analyzeImage = async (file) => {
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence_threshold', 0.01); // Отправляем низкий порог, чтобы получить все результаты

    try {
      const res = await fetch(`${API_PREFIX}/image`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        // Попробуем получить сообщение об ошибке из ответа
        let errorMessage = `Ошибка сервера: ${res.status}`;
        try {
          const errorData = await res.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
          // Если не удалось распарсить JSON ошибки
        }
        
        setError(errorMessage);
        setDetections([]); // Очищаем детекции при ошибке
        return;
      }
      
      const data = await res.json();
      
      // Обрабатываем новый формат ответа с bboxes
      if (data.bboxes && Array.isArray(data.bboxes)) {
        setDetections(data.bboxes);
        // Если бэкенд прислал картинку с разметкой, показываем ее
        if (data.overlay) {
          setPreviewUrl(data.overlay);
        }
      } else if (data.items || data.tools) {
        // Для совместимости со старым форматом
        const items = data.items || data.tools || [];
        const mockDetections = items.map(item => ({
          class: item,
          confidence: 1.0,
          x_min: 0, y_min: 0, x_max: 0, y_max: 0
        }));
        setDetections(mockDetections);
      } else {
        setDetections([]);
      }
      
    } catch (err) {
      console.error('analyze error', err);
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Ошибка подключения к серверу. Проверьте, что бэкенд запущен.');
      } else {
        setError(`Ошибка при обработке изображения: ${err.message}`);
      }
      setDetections([]); // Очищаем детекции при ошибке
    } finally {
      setLoading(false);
    }
  };

  // Удаляем fetchItemsFallback, так как он больше не нужен
  
  // Начать редактирование
  const startEditing = () => {
    const toolsList = createToolsList();
    const initialCounts = {};
    toolsList.forEach(tool => {
      initialCounts[tool.name] = tool.count;
    });
    setEditCounts(initialCounts);
    setIsEditing(true);
  };

  // Сохранить изменения
  const saveEditing = () => {
    const newDetections = [];
    Object.entries(editCounts).forEach(([toolName, count]) => {
      for (let i = 0; i < count; i++) {
        newDetections.push({
          class: toolName,
          confidence: 1.0,
          x_min: 0, y_min: 0, x_max: 0, y_max: 0
        });
      }
    });
    setDetections(newDetections);
    setIsEditing(false);
    setEditCounts({});
  };

  // Отменить редактирование
  const cancelEditing = () => {
    setIsEditing(false);
    setEditCounts({});
  };

  // Изменить количество инструмента
  const updateToolCount = (toolName, newCount) => {
    const count = Math.max(0, parseInt(newCount) || 0);
    setEditCounts(prev => ({
      ...prev,
      [toolName]: count
    }));
  };

  const onConfirm = () => {
    const toolsList = createToolsList();
    const detectedTools = toolsList.filter(tool => tool.count > 0);
    const message = detectedTools.length > 0 
      ? detectedTools.map(tool => `${tool.name}: ${tool.count} шт.`).join('\n')
      : 'Инструменты не обнаружены';
    alert('Подтверждённые инструменты:\n' + message);
  };

  // Функция для очистки изображения
  const clearImage = () => {
    setImageFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setError(null);
    setDetections([]);
    setIsEditing(false);
    setEditCounts({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Создаем объединенный список: дефолтные инструменты + найденные объекты
  const createToolsList = () => {
    // Если в режиме редактирования, используем editCounts
    const countsToUse = isEditing ? editCounts : {};
    
    if (!isEditing) {
      // Группируем детекции по классам и считаем количество
      const detectionCounts = filteredDetections.reduce((acc, detection) => {
        const className = detection.class;
        if (!acc[className]) {
          acc[className] = { count: 0, maxConfidence: 0 };
        }
        acc[className].count++;
        acc[className].maxConfidence = Math.max(acc[className].maxConfidence, detection.confidence);
        return acc;
      }, {});
      
      Object.entries(detectionCounts).forEach(([className, data]) => {
        countsToUse[className] = data.count;
      });
    }

    // Создаем список для отображения
    const toolsList = [];

    // Добавляем дефолтные инструменты с их количеством
    DEFAULT_TOOLS.forEach((tool, index) => {
      const count = countsToUse[tool] || 0;
      let confidence = 0;
      
      if (!isEditing) {
        const detectionCounts = filteredDetections.reduce((acc, detection) => {
          const className = detection.class;
          if (!acc[className]) {
            acc[className] = { count: 0, maxConfidence: 0 };
          }
          acc[className].count++;
          acc[className].maxConfidence = Math.max(acc[className].maxConfidence, detection.confidence);
          return acc;
        }, {});
        confidence = detectionCounts[tool]?.maxConfidence || 0;
      }
      
      toolsList.push({
        name: tool,
        count: count,
        confidence: confidence,
        isDefault: true,
        order: index + 1
      });
    });

    // Добавляем найденные объекты, которых нет в дефолтном списке
    if (!isEditing) {
      Object.entries(countsToUse).forEach(([className, count]) => {
        if (!DEFAULT_TOOLS.includes(className) && count > 0) {
          toolsList.push({
            name: className,
            count: count,
            confidence: 1.0,
            isDefault: false,
            order: 999
          });
        }
      });
    }

    return toolsList.sort((a, b) => a.order - b.order);
  };

  const toolsList = createToolsList();
  const totalDetected = filteredDetections.length; // Используем отфильтрованные
  const totalToolsWithDetections = toolsList.filter(tool => tool.count > 0).length;

  return (
    <div className="processing-app">
      <div className="processing-card">
        <div className="processing-left" onDrop={handleDrop} onDragOver={handleDragOver}>
          <div className="processing-dropArea" onDrop={handleDrop} onDragOver={handleDragOver}>
            {previewUrl ? (
              <img src={previewUrl} alt="preview" className="processing-preview" />
            ) : (
              <div className="processing-placeholder">
                <p>Перетащите изображение сюда</p>
                <p>или</p>
                <p>Кликните, чтобы выбрать файл</p>
                <p style={{ fontSize: '14px', opacity: 0.7 }}>
                  Поддерживаемые форматы: JPG, PNG, GIF, WebP
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="processing-fileInput"
              onChange={handleFileSelect}
            />
          </div>
          
          <div className="processing-helpText">
            {imageFile ? (
              <>
                <div>Файл: {imageFile.name}</div>
                <div>Размер: {(imageFile.size / 1024).toFixed(1)} KB</div>
                <div>Тип: {imageFile.type}</div>
                <div style={{ marginTop: '8px' }}>
                  <button 
                    onClick={clearImage}
                    style={{
                      padding: '4px 8px',
                      fontSize: '12px',
                      background: '#ff6b6b',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    Очистить
                  </button>
                </div>
              </>
            ) : (
              <div>Файл не выбран (макс. 10MB)</div>
            )}
          </div>
        </div>
 
        <div className="processing-right">
          <h3>Обнаруженные объекты</h3>

          {/* Слайдер для порога */}
          <div className="processing-threshold-slider">
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
            />
          </div>

          <div className="processing-toolsBox">
            {loading ? (
              <div style={{textAlign: 'center', color: '#F85A40', marginBottom: '16px'}}>
                <div>Анализ изображения...</div>
                <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.7 }}>
                  Это может занять несколько секунд
                </div>
              </div>
            ) : null}
            
            {/* ВСЕГДА показываем список инструментов */}
            <ul className="processing-list">
              {toolsList.map((tool, index) => (
                <li 
                  key={tool.name} 
                  className="processing-listItem"
                  style={{
                    opacity: tool.count > 0 ? 1 : 0.6,
                    background: tool.count > 0 ? 'rgba(72, 187, 120, 0.1)' : 'transparent',
                    border: tool.count > 0 ? '1px solid rgba(72, 187, 120, 0.3)' : '1px solid transparent',
                    borderRadius: '4px',
                    padding: '6px 8px',
                    marginBottom: '2px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>
                      {tool.isDefault ? `${tool.order}` : '•'} {tool.name}
                    </span>
                    <div style={{ fontSize: '12px', opacity: 0.8, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {isEditing ? (
                        <input
                          type="number"
                          min="0"
                          value={editCounts[tool.name] || 0}
                          onChange={(e) => updateToolCount(tool.name, e.target.value)}
                          style={{
                            width: '50px',
                            padding: '2px 4px',
                            border: '1px solid #ccc',
                            borderRadius: '4px',
                            background: 'white',
                            color: 'black',
                            textAlign: 'center'
                          }}
                        />
                      ) : (
                        <span style={{ 
                          background: tool.count > 0 ? '#48bb78' : '#e2e8f0',
                          color: tool.count > 0 ? 'white' : '#4a5568',
                          padding: '2px 6px',
                          borderRadius: '12px',
                          fontWeight: 'bold',
                          minWidth: '20px',
                          textAlign: 'center'
                        }}>
                          {tool.count}
                        </span>
                      )}
                      {!isEditing && tool.count > 0 && (
                        <span>{(tool.confidence * 100).toFixed(1)}%</span>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
            
            {error && <div className="processing-error">{error}</div>}
            
            {!loading && (
              <div style={{ 
                marginTop: '12px', 
                padding: '8px', 
                background: 'rgba(255,255,255,0.1)', 
                borderRadius: '6px',
                fontSize: '12px',
                opacity: 0.8
              }}>
                Найдено инструментов: {totalToolsWithDetections} из {DEFAULT_TOOLS.length}
                {totalDetected > 0 && (
                  <>
                    <br />
                    Всего объектов на изображении: {totalDetected}
                  </>
                )}
              </div>
            )}
          </div>

          <div className="processing-buttonsRow">
            {isEditing ? (
              <>
                <button 
                  className="processing-button" 
                  onClick={saveEditing}
                  style={{ background: '#48bb78' }}
                >
                   Сохранить
                </button>
                <button 
                  className="processing-button" 
                  onClick={cancelEditing}
                  style={{ background: '#e53e3e' }}
                >
                   Отмена
                </button>
              </>
            ) : (
              <>
                <button className="processing-button" onClick={startEditing}> Редактировать</button>
                <button className="processing-button" onClick={onConfirm}> ОК</button>
              </>
            )}
          </div>
          
          <div style={{ marginTop: "20px", textAlign: "center" }}>
            <button className="processing-button" onClick={() => window.location.href = "/"}>
              ⬅ Назад в меню
            </button>
          </div>
          
          {/* <div className="processing-backendInfo">
            <div> Бэкенд: {BACKEND.replace('/api', '')}</div>
            <div> Бэкенд: {API_PREFIX.replace('/api/v1/infer', '')}</div>
            <div>
               API:
              <div className="processing-apiExample">
                POST /api/v1/infer/image (FormData) → {`{ bboxes: [...] }`}
              </div>
              <div style={{ fontSize: '10px', marginTop: '4px', opacity: 0.6 }}>
                Формат: {`{ file: "data:image/..." }`}
              </div>
            </div>
          </div> */}
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