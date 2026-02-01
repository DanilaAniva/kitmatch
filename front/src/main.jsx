import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MainMenu from "./pages/MainMenu.jsx";
import SingleImagePage from "./pages/SingleImagePage.jsx";
import BatchPage from "./pages/BatchPage.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainMenu />} />
        <Route path="/single" element={<SingleImagePage />} />
        <Route path="/batch" element={<BatchPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
