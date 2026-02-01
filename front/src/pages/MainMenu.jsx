import React from "react";
import { Link } from "react-router-dom";

export default function MainMenu() {
  return (
    <div className="main-menu-app">
      <div className="main-menu-card">
        {/* Логотип Аэрофлота вверху */}
        <img 
          src="/assets/aeroflot_logo.png" 
          alt="Аэрофлот" 
          className="aeroflot-logo"
        />
        
        <h1 className="main-title">Обработка изображений</h1>
        <p className="subtitle">
          Выберите тип обработки изображений для вашего проекта
        </p>
        <div className="menu-buttons">
          <Link to="/single" className="menu-link">
            <button className="menu-button">
               Обработка одного изображения
            </button>
          </Link>
          <Link to="/batch" className="menu-link">
            <button className="menu-button">
               Обработка архива изображений
            </button>
          </Link>
        </div>
      </div>
      
      {/* Логотип ЛЦТ снизу под карточкой */}
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