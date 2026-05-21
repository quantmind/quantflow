const apiOrigin = process.env.QUANTFLOW_API_ORIGIN || "";

export default {
  title: "Quantflow Examples",
  root: "src",
  output: "../examples",
  base: "/examples",
  head: `<meta name="quantflow-api-origin" content="${apiOrigin}">`,
  style: "style.css",
  pages: [{name: "Volatility Surface", path: "/volatility-surface"}, {name: "Yield Curve", path: "/yield-curve"}, {name: "Sampling", path: "/sampling"}, {name: "SuperSmoother", path: "/supersmoother"}, {name: "Cointegration", path: "/cointegration"}, {name: "Hurst Exponent", path: "/hurst"}, {name: "Heston Vol Surface", path: "/heston-vol-surface"}],
  header: `
    <nav class="qf-header-inner">
      <a href="https://quantflow.quantmind.com" class="qf-header-logo" title="QuantFlow">
        <img src="https://quantflow.quantmind.com/assets/logos/quantflow-mark-dark.svg" alt="logo">
        <span class="qf-header-title">QuantFlow</span>
      </a>
      <div class="qf-header-spacer"></div>
      <a class="qf-header-link" href="https://github.com/quantmind/quantflow" title="Go to repository">quantmind/quantflow</a>
    </nav>
  `,
  footer: "Quantflow live examples"
};
