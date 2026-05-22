import {readFileSync} from "node:fs";
import {resolve, dirname} from "node:path";
import {fileURLToPath} from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const apiOrigin = process.env.QUANTFLOW_API_ORIGIN || "";
const headSnippet = readFileSync(resolve(__dirname, "../../docs/assets/logos/head-snippet.html"), "utf-8")
  .trim()
  .replace(/href="\//g, 'href="https://quantflow.quantmind.com/');

export default {
  title: "Quantflow Examples",
  root: "src",
  output: "../app/examples",
  base: "/examples",
  head: `<meta name="quantflow-api-origin" content="${apiOrigin}">\n${headSnippet}`,
  style: "style.css",
  pages: [{name: "Volatility Surface", path: "/volatility-surface"}, {name: "Yield Curve", path: "/yield-curve"}, {name: "Sampling", path: "/sampling"}, {name: "SuperSmoother", path: "/supersmoother"}, {name: "Cointegration", path: "/cointegration"}, {name: "Hurst Exponent", path: "/hurst"}, {name: "Heston Vol Surface", path: "/heston-vol-surface"}],
  header: `
    <nav class="qf-header-inner">
      <a href="https://quantflow.quantmind.com" class="qf-header-logo" title="QuantFlow">
        <img src="https://quantflow.quantmind.com/assets/logos/quantflow-mark-dark.svg" alt="logo">
        <span class="qf-header-title">QuantFlow</span>
      </a>
      <div class="qf-header-spacer"></div>
      <button class="qf-theme-toggle" id="qf-theme-toggle" title="Toggle light/dark theme" aria-label="Toggle theme">
        <svg class="qf-icon-light" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7m2 14H10v-1h4v1m0-2H10v-1h4v1m-1.5 5h-1c-.28 0-.5-.22-.5-.5s.22-.5.5-.5h1c.28 0 .5.22.5.5s-.22.5-.5.5m1-1h-3c-.28 0-.5-.22-.5-.5s.22-.5.5-.5h3c.28 0 .5.22.5.5s-.22.5-.5.5"/></svg>
        <svg class="qf-icon-dark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.87-3.13-7-7-7m2 14H10v-1h4v1m0-2H10v-1h4v1m-1.5 5h-1c-.28 0-.5-.22-.5-.5s.22-.5.5-.5h1c.28 0 .5.22.5.5s-.22.5-.5.5m1-1h-3c-.28 0-.5-.22-.5-.5s.22-.5.5-.5h3c.28 0 .5.22.5.5s-.22.5-.5.5M12 4c2.76 0 5 2.24 5 5 0 1.94-1.11 3.61-2.73 4.44l-.27.14V15h-4v-1.42l-.27-.14A4.997 4.997 0 0 1 7 9c0-2.76 2.24-5 5-5"/></svg>
      </button>
      <a class="qf-header-link" href="https://github.com/quantmind/quantflow" title="Go to repository">quantmind/quantflow</a>
    </nav>
    <script>
      (function() {
        var root = document.documentElement;
        var stored = localStorage.getItem("qf-theme");
        if (stored) root.setAttribute("data-theme", stored);
        var btn = document.getElementById("qf-theme-toggle");
        btn.addEventListener("click", function() {
          var current = root.getAttribute("data-theme");
          var next = current === "light" ? "dark" : "light";
          root.setAttribute("data-theme", next);
          localStorage.setItem("qf-theme", next);
        });
      })();
    </script>
  `,
  footer: "Quantflow live examples"
};
