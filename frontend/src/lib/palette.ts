import * as d3 from "npm:d3";

export const sample = (f: (t: number) => string, n = 10): string[] =>
  Array.from({length: n}, (_, i) => f(i / (n - 1)));

export const schemes: Record<string, readonly string[]> = {
  // Categorical
  "Category10":   d3.schemeCategory10,
  "Observable10": d3.schemeObservable10,
  "Tableau10":    d3.schemeTableau10,
  "Accent":       d3.schemeAccent,
  "Dark2":        d3.schemeDark2,
  "Paired":       d3.schemePaired,
  "Pastel1":      d3.schemePastel1,
  "Pastel2":      d3.schemePastel2,
  "Set1":         d3.schemeSet1,
  "Set2":         d3.schemeSet2,
  "Set3":         d3.schemeSet3,
  // Sequential — single hue
  "Blues":         sample(d3.interpolateBlues),
  "Greens":       sample(d3.interpolateGreens),
  "Greys":        sample(d3.interpolateGreys),
  "Oranges":      sample(d3.interpolateOranges),
  "Purples":      sample(d3.interpolatePurples),
  "Reds":         sample(d3.interpolateReds),
  // Sequential — multi hue
  "Turbo":        sample(d3.interpolateTurbo),
  "Viridis":      sample(d3.interpolateViridis),
  "Inferno":      sample(d3.interpolateInferno),
  "Magma":        sample(d3.interpolateMagma),
  "Plasma":       sample(d3.interpolatePlasma),
  "Cividis":      sample(d3.interpolateCividis),
  "Warm":         sample(d3.interpolateWarm),
  "Cool":         sample(d3.interpolateCool),
  "Cubehelix":    sample(d3.interpolateCubehelixDefault),
  "BuGn":         sample(d3.interpolateBuGn),
  "BuPu":         sample(d3.interpolateBuPu),
  "GnBu":         sample(d3.interpolateGnBu),
  "OrRd":         sample(d3.interpolateOrRd),
  "PuBuGn":       sample(d3.interpolatePuBuGn),
  "PuBu":         sample(d3.interpolatePuBu),
  "PuRd":         sample(d3.interpolatePuRd),
  "RdPu":         sample(d3.interpolateRdPu),
  "YlGnBu":       sample(d3.interpolateYlGnBu),
  "YlGn":         sample(d3.interpolateYlGn),
  "YlOrBr":       sample(d3.interpolateYlOrBr),
  "YlOrRd":       sample(d3.interpolateYlOrRd),
  // Diverging
  "BrBG":         sample(d3.interpolateBrBG),
  "PRGn":         sample(d3.interpolatePRGn),
  "PiYG":         sample(d3.interpolatePiYG),
  "PuOr":         sample(d3.interpolatePuOr),
  "RdBu":         sample(d3.interpolateRdBu),
  "RdGy":         sample(d3.interpolateRdGy),
  "RdYlBu":       sample(d3.interpolateRdYlBu),
  "RdYlGn":       sample(d3.interpolateRdYlGn),
  "Spectral":     sample(d3.interpolateSpectral),
  // Cyclical
  "Rainbow":      sample(d3.interpolateRainbow),
  "Sinebow":      sample(d3.interpolateSinebow),
};

interface ThemePalette {
  dark: string;
  light: string;
}

export const palette1: ThemePalette = {dark: "Observable10", light: "Dark2"};
export const palette2: ThemePalette = {dark: "viridis", light: "cividis"};
export const palette3: ThemePalette = {dark: "cool", light: "YlGnBu"};

function getTheme(): "light" | "dark" {
  return document.documentElement.getAttribute("data-theme") === "light"
    ? "light"
    : "dark";
}

export function currentScheme(config: ThemePalette = palette1): string {
  return config[getTheme()];
}

/**
 * Observable generator that emits the active scheme name
 * whenever the theme toggle changes.
 */
export function observeScheme(config: ThemePalette = palette1) {
  return (notify: (value: string) => void) => {
    const update = () => notify(config[getTheme()]);
    update();
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.attributeName === "data-theme") {
          update();
          break;
        }
      }
    });
    observer.observe(document.documentElement, {attributes: true});
    return () => observer.disconnect();
  };
}
