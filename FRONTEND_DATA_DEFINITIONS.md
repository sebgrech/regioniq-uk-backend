# Frontend Data Definitions

## Data Sources Overview

### Annual Population Survey (APS)
The **Annual Population Survey (APS)** is a continuous household survey conducted by the Office for National Statistics (ONS). It collects information about the labour market status and personal characteristics of people living in the UK.

**Key characteristics:**
- **Geography**: Residence-based (where people live)
- **Method**: Survey-based (sample of households)
- **Coverage**: All UK residents aged 16 and over
- **Frequency**: Quarterly data, published annually
- **Used for**: Employment rate, unemployment rate

### Business Register and Employment Survey (BRES)
The **Business Register and Employment Survey (BRES)** is an annual survey of businesses conducted by the ONS. It collects information about the number of jobs (employee and self-employed jobs) at each business location.

**Key characteristics:**
- **Geography**: Workplace-based (where jobs are located)
- **Method**: Survey-based (sample of businesses)
- **Coverage**: All employee and self-employed jobs in Great Britain (excludes Northern Ireland)
- **Frequency**: Annual
- **Used for**: Total employment (jobs)

**Important difference**: APS measures people (residence-based), while BRES measures jobs (workplace-based). This means:
- A person who lives in Area A but works in Area B is counted in Area A's employment rate (APS) but Area B's total jobs (BRES)
- At the UK level, these match closely (~3 percentage points difference)
- At local authority level, differences can be significant due to commuting (e.g., Westminster has 400+ percentage points difference)

---

## Metric Definitions

### Population (Total)
**Definition**: The total number of people living in the area, regardless of age.

**Source**: NOMIS dataset NM_2002_1 (ONS Mid-Year Population Estimates)

**Calculation**: 
- All ages (c_age=200)
- All genders
- Residence-based (where people live)

**Unit**: Persons

**Note**: This includes all residents, from newborns to the oldest members of the population.

---

### Population (16-64) - Working Age Population
**Definition**: The number of people aged 16 to 64 years old living in the area. This is the "working age" population used as the denominator for employment and unemployment rates.

**Source**: NOMIS dataset NM_2002_1 (ONS Mid-Year Population Estimates)

**Calculation**: 
- Ages 16-64 inclusive (c_age=203)
- All genders
- Residence-based (where people live)

**Unit**: Persons

**Note**: This is the standard working age population used by ONS for labour market statistics. It excludes people under 16 and those aged 65 and over.

---

### Employment Rate
**Definition**: The percentage of people aged 16-64 who are in employment (employed or self-employed).

**Source**: NOMIS dataset NM_17_5 (Annual Population Survey - APS)

**Calculation**: 
```
Employment Rate (%) = (Employed people aged 16-64 / Population aged 16-64) × 100
```

**Geography**: Residence-based (where people live)

**Unit**: Percent (%)

**Important notes**:
- This is a **residence-based** rate (where people live, not where they work)
- It measures the proportion of working-age residents who are employed
- It does **not** equal `(Total Jobs / Population 16-64) × 100` because:
  - Employment rate uses APS (residence-based)
  - Total jobs uses BRES (workplace-based)
  - People may live in one area but work in another (commuting)
- At UK level, the difference is small (~3pp) because there is no net commuting
- At local authority level, differences can be large in areas with high commuting (e.g., city centres, commuter towns)

**Example**: If 75% of working-age residents in an area are employed, the employment rate is 75%, regardless of where those people work.

---

### Unemployment Rate
**Definition**: The percentage of economically active people aged 16 and over who are unemployed.

**Source**: NOMIS dataset NM_17_5 (Annual Population Survey - APS)

**Calculation**: 
```
Unemployment Rate (%) = (Unemployed people aged 16+ / Economically active people aged 16+) × 100
```

Where:
- **Economically active** = Employed + Unemployed (those who are either working or actively seeking work)
- **Unemployed** = People who are not in employment but are actively seeking work and available to start work

**Geography**: Residence-based (where people live)

**Unit**: Percent (%)

**Important notes**:
- This is a **residence-based** rate
- The denominator is economically active people (not total population)
- People who are not economically active (e.g., students, retirees, long-term sick) are excluded from both numerator and denominator
- This follows the International Labour Organization (ILO) definition of unemployment

---

### Total Employment (Jobs)
**Definition**: The total number of employee and self-employed jobs located in the area.

**Source**: NOMIS datasets NM_172_1 (2009-2015) and NM_189_1 (2015-2024) (Business Register and Employment Survey - BRES)

**Geography**: Workplace-based (where jobs are located)

**Unit**: Jobs

**Important notes**:
- This is a **workplace-based** count (where jobs are located, not where workers live)
- Includes both employee jobs and self-employed jobs
- A person with two jobs counts as two jobs
- Only available for Great Britain (excludes Northern Ireland)
- This does **not** equal `(Employment Rate / 100) × Population 16-64` because:
  - Total jobs uses BRES (workplace-based)
  - Employment rate uses APS (residence-based)
  - People may live in one area but work in another (commuting)

**Example**: If 100,000 jobs are located in an area, that's the total employment count, regardless of where the workers live.

---

### Gross Value Added (GVA)
**Definition**: Gross value added (GVA) is the value generated by any economic unit that produces goods and services. It reflects the value of goods and services produced, less the cost of any inputs used up in that production process. GVA is a standard measure of the economic activity taking place in an area. It comprises the majority of gross domestic product (GDP), only excluding taxes and subsidies (such as Value Added Tax and duty on fuel or alcohol).

**Source**: NOMIS dataset NM_2400_1 (ONS Regional Accounts)

**Calculation**: 
- GVA is measured at current prices (nominal)
- GVA for the UK is measured by the UK National Accounts and published each year in the annual Blue Book
- The GVA is then broken down to individual countries, regions, and local authority districts

**Unit**: Millions GBP (£ millions)

**Important notes**:
- GVA measures the economic output of an area
- It represents the value added by production activities (output minus intermediate consumption)
- GVA + taxes on products - subsidies on products = GDP
- GVA is a key indicator of regional economic performance
- Higher GVA indicates greater economic activity and productivity in an area

**For more information**: Regional Accounts, Office for National Statistics  
Email: regionalaccounts@ons.gov.uk  
Tel: +44(0) 1633 456878

---

### Gross Disposable Household Income (GDHI) per Head
**Definition**: Gross disposable household income (GDHI) is the amount of money that all of the individuals in the household sector have available for spending or saving after income distribution measures (for example, taxes, social contributions and benefits) have taken effect. GDHI is a concept which is seen to reflect the 'material welfare' of the household sector.

**Calculation**:
```
GDHI per Head = (GDHI Total in £ millions × 1,000,000) / Total Population
```

**Source**: NOMIS dataset NM_185_1 (ONS Regional Accounts)

**Unit**: GBP (£) per person

**Important notes**:
- Regional GDHI estimates relate to totals for all individuals within the household sector for a region rather than to an average household or family unit
- The household sector comprises all individuals in an economy, including people living in traditional households as well as those living in institutions such as retirement homes and prisons
- The sector also includes sole trader enterprises (the self-employed)
- GDHI per head is calculated by dividing total GDHI by total population (all ages)

**For more information**: Regional Accounts, Office for National Statistics  
Email: regionalaccounts@ons.gov.uk  
Tel: +44(0) 1633 456878

---

## Key Differences Summary

| Metric | Source | Geography | What It Measures |
|--------|--------|-----------|------------------|
| Employment Rate | APS | Residence | % of working-age residents who are employed |
| Unemployment Rate | APS | Residence | % of economically active residents who are unemployed |
| Total Employment (Jobs) | BRES | Workplace | Number of jobs located in the area |
| Population (Total) | ONS Mid-Year Estimates | Residence | Total number of residents |
| Population (16-64) | ONS Mid-Year Estimates | Residence | Number of working-age residents |
| GVA | ONS Regional Accounts | Workplace | Value of economic output in the area |
| GDHI per Head | ONS Regional Accounts | Residence | Average disposable income per resident |

**Why Employment Rate ≠ (Total Jobs / Population 16-64) × 100:**
- Employment rate is residence-based (where people live)
- Total jobs is workplace-based (where jobs are located)
- Commuting creates differences between these measures
- At UK level: difference is small (~3pp)
- At local level: differences can be large in areas with high commuting

---

## Data Quality Indicators

The `data_quality` column indicates the source and method of the data:

- **`ONS`**: Original data from Office for National Statistics (no modifications)
- **`interpolated`**: Data point was filled using linear interpolation between two known ONS values (gap-filling for missing years between known data points)
- **`NULL`** (or missing): Forecast data (years beyond the last available ONS data point)

**Note**: Interpolated values are only used for gaps between known ONS years. Years beyond the last available ONS year are handled by the forecasting system and marked as forecasts.

---

## Contact Information

For questions about these datasets, please contact:

**Regional Accounts, Office for National Statistics**  
Email: regionalaccounts@ons.gov.uk  
Tel: +44(0) 1633 456878

