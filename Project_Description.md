# Raw Material Delivery Forecasting — Task Specification

## Context

Consultancy project by **Append Consulting** for **Hydro** (Norwegian industrial company in energy, aluminum, and recycling). The forecasts produced here feed into a larger optimization tool developed by Append.

## Objective

Develop **accurate but conservative** forecasts of incoming raw material deliveries.

For each raw material (identified by `rm_id`), predict the **cumulative weight (kg) of incoming deliveries** from **January 1, 2025** up to and including a specified end date in the range **Jan 1 – May 31, 2025**.

Historical data is provided through the end of 2024.

## Evaluation Metric

Asymmetric **Quantile (Pinball) Loss at τ = 0.2** applied to cumulative deliveries over the forecast window.

For each raw material `i` and horizon `h` days after time `T`:

- Actual cumulative delivery: `A_i = Σ_{t=1..h} y_{i, T+t}`
- Forecast cumulative delivery: `F_i = Σ_{t=1..h} ŷ_{i, T+t}`

Per-material loss:

```
QuantileLoss_0.2(F_i, A_i) = max( 0.2 * (A_i - F_i),  0.8 * (F_i - A_i) )
```

Final score (lower is better):

```
QuantileError_0.2 = (1/N) * Σ_i QuantileLoss_0.2(F_i, A_i)
```

### Key implication for modeling

- **Over-prediction is penalized 4× more than under-prediction** (0.8 vs 0.2 per unit).
- The optimal point forecast is the **0.2-quantile** of the predictive distribution of cumulative deliveries — i.e. systematically biased low (conservative).
- Do **not** predict the mean or median; predict the 20th percentile.

## Submission Format

A CSV with **exactly two columns**: `ID` and `predicted_weight`.

Use `data/prediction_mapping.csv` to construct the submission — it maps each `ID` to the `(rm_id, end_date)` pair you need to forecast for.

```
ID,predicted_weight
1,0
2,0
3,0
...
```

`predicted_weight` is the cumulative kg delivered for that `rm_id` from Jan 1, 2025 through the corresponding end date (inclusive).

Reference: `sample_submission.csv` in the data folder.

## Datasets

Organized into two folders: `kernel/` (core) and `extended/` (optional metadata).

### `data/kernel/receivals.csv` — primary dataset, **required**

Historical records of material receivals. The target variable is built from `net_weight` aggregated by `rm_id` and `date_arrival`.

| Column | Description |
|---|---|
| `rm_id` | Unique raw material identifier (the entity being forecast) |
| `product_id` | Specific product received; each `rm_id` can have multiple `product_id`s |
| `purchase_order_id` | Links to the corresponding purchase order |
| `purchase_order_item_no` | Links to the corresponding purchase order item |
| `batch_id` | Each purchase can be split into multiple batches |
| `receival_item_no` | Each PO item can be split into several receivals; this identifies them |
| `date_arrival` | **UTC timestamp of arrival — use this to assign the receival to a date** |
| `receival_status` | Text status (e.g. "Completed"). **All statuses count toward the target.** |
| `net_weight` | Weight of product excluding packaging — **basis for the target variable** |
| `supplier_id` | Supplier for the purchase |

### `data/kernel/purchase_orders.csv` — **strongly recommended**

Ordered quantities and expected delivery dates. Useful as leading indicator features.

| Column | Description |
|---|---|
| `purchase_order_id` | Joinable with receivals and transportation |
| `purchase_order_item_no` | Joinable with receivals and transportation |
| `quantity` | Amount ordered |
| `delivery_date` | Expected delivery date (sometimes placed at month/year-end as a placeholder) |
| `product_id` | Product ordered (Hydro orders by `product_id`, later assigns `rm_id` on receival) |
| `product_version` | Version/subtype |
| `created_date_time` | When the record was created |
| `modified_date_time` | Last edit timestamp |
| `unit_id`, `unit` | Unit (typically kg) |
| `status_id`, `status` | Status (e.g. "Closed") |

**Note:** orders specify `product_id`, not `rm_id`. The `rm_id` is determined later at receival time, so the order→rm_id link is not direct.

### `data/extended/materials.csv` — optional

Metadata on raw materials.

| Column | Description |
|---|---|
| `rm_id` | Raw material identifier |
| `product_id` | Product identifier |
| `product_version` | Product version |
| `raw_material_alloy` | Alloy name |
| `raw_material_format_type` | Physical form (block, powder, etc.) |
| `stock_location` | Storage/warehouse position |

### `data/extended/transportation.csv` — optional

Transportation data that may affect delivery timing and consistency.

## BIG HINT

The Presentations-folder contains presentation of the top 3 teams for this competition. They present what methods they used in order to achieve good results. You should use these presentations as a big help when designing your model.

## Modeling Notes / Hints

- The target is a **cumulative sum** over a window starting Jan 1, 2025 — for any given `rm_id`, predictions for later end dates must be **monotonically non-decreasing** in the end date.
- A correct submission file will contain a lot of zero-values for predictions. This is because there are many materials and a lot of them will not be ordered. The submission file will be sparse.