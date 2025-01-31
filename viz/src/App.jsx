import { PanelDisplay, useModelData } from '@nextstrain/evofr-viz';
import '@nextstrain/evofr-viz/dist/index.css';

const customAddress = !!import.meta.env.VITE_DATA_HOST;
const mlrCladesConfig = {
    modelName: "mlr_clades",
    modelUrl: customAddress ?
      `${import.meta.env.VITE_DATA_HOST}/${import.meta.env.VITE_CLADES_PATH}` :
      `https://nextstrain-data.s3.amazonaws.com/files/workflows/forecasts-ncov/gisaid/nextstrain_clades/global/mlr/latest_results.json`,
}
const mlrLineagesConfig = {
    modelName: "mlr_lineages",
    modelUrl: customAddress ?
      `${import.meta.env.VITE_DATA_HOST}/${import.meta.env.VITE_LINEAGES_PATH}` :
      `https://nextstrain-data.s3.amazonaws.com/files/workflows/forecasts-ncov/gisaid/pango_lineages/global/mlr/latest_results.json`,
}

function App() {

  const mlrCladesData = useModelData(mlrCladesConfig);
  const mlrLineagesData = useModelData(mlrLineagesConfig);

  const mlrCladesPivotRaw = mlrCladesData?.modelData?.get('pivot') || "loading";
  const mlrCladesPivot = mlrCladesData?.modelData?.get('variantDisplayNames')?.get(mlrCladesPivotRaw) || mlrCladesPivotRaw;
  const mlrLineagesPivot = mlrLineagesData?.modelData?.get('pivot') || "loading";

  const cladesLocationsFiltered = mlrCladesData?.modelData?.get('locations')?.filter((loc)=>loc!=='hierarchical') || [];
  const lineagesLocationsFiltered = mlrLineagesData?.modelData?.get('locations')?.filter((loc)=>loc!=='hierarchical') || [];

  return (
    <div className="App">

      <div id="mainPanelsContainer">
        <h2>Clade frequencies over time</h2>
        <p>
          Each line represents the estimated frequency of a particular clade through time.
          Equivalent Pango lineage is given in parenthesis, eg clade 23A (lineage XBB.1.5). Only
          locations with more than 1000 sequences from samples collected in the previous 150 days are
          included. Results last updated {mlrCladesData?.modelData?.get('updated') || 'loading'}.
        </p>
        <div id="cladeFrequenciesPanel" class="panelDisplay"> {/* surrounding div(s) used for static-images.js script */}
          <PanelDisplay data={mlrCladesData} locations={cladesLocationsFiltered} params={{preset: "frequency"}}/>
        </div>

        <h2>Clade growth advantage</h2>
        <p>
          These plots show the estimated growth advantage for given clades relative to
          clade {mlrCladesPivot}. This describes how many more secondary infections a variant causes
          on average relative to clade {mlrCladesPivot}. Vertical bars show the 95% HPD. The "hierarchical" panel
          shows pooled estimate of growth rates across different locations.
          Results last updated {mlrCladesData?.modelData?.get('updated') || 'loading'}.
        </p>
        <div id="cladeGrowthAdvantagePanel" class="panelDisplay">
          <PanelDisplay data={mlrCladesData} params={{preset: "growthAdvantage"}}/>
        </div>

        <h2>Lineage frequencies over time</h2>
        <p>
          Each line represents the estimated frequency of a particular Pango lineage through time.
          Lineages with fewer than 350 observations are collapsed into parental lineage. Only
          locations with more than 1000 sequences from samples collected in the previous 150 days are
          included. Results last updated {mlrLineagesData?.modelData?.get('updated') || 'loading'}.
        </p>
        <div id="lineageFrequenciesPanel" class="panelDisplay">
          <PanelDisplay data={mlrLineagesData} locations={lineagesLocationsFiltered} params={{preset: "frequency"}}/>
        </div>

        <h2>Lineage growth advantage</h2>
        <p>
          These plots show the estimated growth advantage for given Pango lineages relative to
          lineage {mlrLineagesPivot}. This describes how many more secondary infections a variant causes
          on average relative to lineage {mlrLineagesPivot}. Vertical bars show the 95% HPD.
          The "hierarchical" panel shows pooled estimate of growth rates across different locations.
          Results last updated {mlrLineagesData?.modelData?.get('updated') || 'loading'}.
        </p>
        <div id="lineageGrowthAdvantagePanel" class="panelDisplay">
          <PanelDisplay data={mlrLineagesData} params={{preset: "growthAdvantage"}}/>
        </div>

      </div>
    </div>
  )
}

export default App
