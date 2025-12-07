# Frontend Data Quality Tooltip Guide

## Overview

The `data_quality` column in Supabase indicates the provenance of historical data points:

- `ONS`: Original ONS data (observed)
- `estimated`: Gap-filled data (interpolated or inherited from parent ITL3)
- `NULL`: Forecast data (always NULL for forecasts)

## Recharts Tooltip Example

Here's how to show data quality in Recharts tooltips:

```typescript
import { LineChart, Line, Tooltip, XAxis, YAxis } from 'recharts';

// Custom tooltip component
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const dataQuality = data.data_quality;
    
    // Determine quality label
    let qualityLabel = '';
    let qualityColor = '';
    
    if (dataQuality === 'interpolated') {
      qualityLabel = 'Interpolated';
      qualityColor = '#f59e0b'; // amber
    } else if (dataQuality === 'ONS') {
      qualityLabel = 'ONS';
      qualityColor = '#10b981'; // green
    }
    
    return (
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
        <p className="font-semibold">
          {data.period}: {data.value.toFixed(2)}{data.unit || '%'}
        </p>
        {qualityLabel && (
          <p 
            className="text-xs mt-1"
            style={{ color: qualityColor }}
          >
            {qualityLabel}
          </p>
        )}
        {data.data_type === 'forecast' && (
          <p className="text-xs text-gray-500 mt-1">Forecast</p>
        )}
      </div>
    );
  }
  return null;
};

// Usage in chart
<LineChart data={chartData} width={800} height={400}>
  <XAxis dataKey="period" />
  <YAxis />
  <Tooltip content={<CustomTooltip />} />
  <Line 
    type="monotone" 
    dataKey="value" 
    stroke="#3b82f6"
    strokeWidth={2}
  />
</LineChart>
```

## Alternative: Visual Indicators

You can also use different line styles or markers for interpolated data:

```typescript
// Different stroke dash for interpolated data
<Line 
  type="monotone" 
  dataKey="value" 
  stroke="#3b82f6"
  strokeWidth={2}
  strokeDasharray={data.data_quality === 'ONS' ? '0' : '5 5'}
/>
```

## Data Quality Values

| Value | Meaning | When Used |
|-------|---------|-----------|
| `ONS` | Original ONS data | All historical data from NOMIS |
| `interpolated` | Gap-filled data | Interpolated or inherited from parent ITL3 |
| `NULL` | Forecast | All forecast data |

## Notes

- `data_quality` is populated for LAD, ITL3, and ITL2 rate metrics (unemployment_rate_pct, employment_rate_pct)
- For ITL3/ITL2, `data_quality` is derived from constituent LADs: if any LAD is `interpolated`, the aggregate is `interpolated`
- For ITL1, `data_quality` defaults to `'ONS'` (separate NOMIS ingest)
- Forecast data always has `data_quality = NULL`

