import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { ThemeProvider } from "./lib/themeMode";
import { NeuralTransitionHost } from "./components/NeuralTransition";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <NeuralTransitionHost>
        <App />
      </NeuralTransitionHost>
    </ThemeProvider>
  </StrictMode>,
);
