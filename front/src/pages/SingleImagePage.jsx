import React from "react";
import { useNavigate } from "react-router-dom";
import App from "../App.jsx";

export default function SingleImagePage() {
  const navigate = useNavigate();

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <App />
    </div>
  );
}