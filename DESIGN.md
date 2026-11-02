# Design

Written and committed before any outcome was estimated. By this point the only trip data
examined was treatment-side: the `cbd_congestion_fee` and `congestion_surcharge` columns for
February 2025, used to learn who is charged. Later changes go in the deviations log at the
end, and nothing above that log is edited after estimation.

## The question

What did the start of congestion pricing on 5 January 2025 do to the number of recorded
for-hire trips touching Manhattan's Congestion Relief Zone (CRZ)?

This is a question about TLC trip records. Yellow taxis and high-volume for-hire vehicles
(Uber, Lyft) are a subset of vehicles. Private cars, trucks, buses, deliveries and black cars
outside the high-volume bases are absent. Nothing here measures total traffic, congestion or
pollution, and no estimate may be described as an effect on any of them.

## What the treatment actually is

Facts checked against primary sources:

| Fact | Value | Source |
|---|---|---|
| Start | Sunday 5 January 2025, midnight | NBC New York citing the MTA; TLC data dictionary ("starting Jan. 5, 2025") |
| Legal boundary | Manhattan south of and inclusive of 60th Street, excluding the FDR Drive, West Side Highway/Route 9A, the Battery Park Underpass, and any surface portion of the Hugh L. Carey Tunnel connecting to West Street | MTA Reform and Traffic Mobility Act, quoted in the data.ny.gov geofence record |
| Passenger car toll | $9 peak, $2.25 overnight, E-ZPass | mta.info |
| Peak hours | 5am-9pm weekdays, 9am-9pm weekends | mta.info |
| Taxi and for-hire charge | Per trip, not daily: $1.50 high-volume FHV, $0.75 yellow, green and other FHV, on trips "to/from/within/through" the zone | mta.info taxi and FHV tolls page |
| `cbd_congestion_fee` | "Per-trip charge for MTA's Congestion Relief Zone starting Jan. 5, 2025." New from 2025 files | TLC dictionaries, 18 March 2025; TLC trip data page |
| Older surcharge | $2.50 taxi, $2.75 FHV on trips touching Manhattan south of 96th Street, since February 2019 | nyc.gov/tlc congestion surcharge page |
| Prior history | Paused 5 June 2024; revived at $9 on 14 November 2024 | Streetsblog, CBS News |

Four consequences shape the design.

**For-hire vehicles do not pay the $9 toll.** They pay a flat per-trip charge that is a small
fraction of a fare, and the same day and night. So the treatment here is a mix of a small price
rise for riders, plus whatever follows from the toll on everyone else: less private traffic may
raise for-hire speed and supply, and people who stop driving may switch to taxis. The direct
price effect pushes trips down and the indirect effects could push them up. The sign is an
empirical question.

**Two charges must be kept apart.** `congestion_surcharge` is the 2019 state surcharge, south of
96th Street, in place through the entire sample. It is not the treatment. The zones between
60th and 96th paid it before and after and pay no CRZ charge, which makes them the closest
comparison on price and, for the same reason, the most exposed to displacement.

**Taxi zones do not follow the boundary.** Overlaying the MTA geofence on the TLC shapefile:

- 20 zones lie entirely inside the geofence.
- 17 more are 79% to 93% inside. The rest of their area is shoreline and excluded highway
  corridor. In February 2025 between 86% and 99% of their trips were charged.
- 7 are genuinely mixed: Lincoln Square East and West, Lenox Hill East and West, Upper East
  Side South, Central Park (1% to 14% overlap, because their southern edge is 59th Street while
  the toll line is 60th) and Battery Park (9% overlap by area but 92% of inbound trips charged:
  geometry alone misclassifies it). Roughly 15% to 40% of their trips are charged.

Rule: overlap of at least 0.75 is `core`; overlap between 0.005 and 0.75 is `mixed`.

**The fee is a direct treatment signal.** Using it gives a first stage that geometry alone
cannot: in February 2025 about 98% of core-touching trips were charged, and 1.2% of trips with
both ends outside core and mixed zones were. Those 1.2% are mostly Newark and out-of-city
trips, which fall outside the comparison units. The pipeline reports the charged share for
every flow, so any leakage is visible rather than assumed away.

## Estimand

The average treatment effect on the treated: the proportional change in weekly recorded
for-hire trips touching core zones, relative to what those trips would have been without the
program, averaged over 5 January 2025 to 26 September 2026.

This is an intent-to-treat effect of the whole program on these trips, not the price response
to a $0.75 or $1.50 charge. The design cannot separate the channels above.

## Unit, outcome and flows

Unit: taxi zone by week. Weeks run Sunday to Saturday, numbered so week 0 starts on 5 January
2025 and no week straddles the start.

The outcome is the count of trips, combining yellow and high-volume records, after dropping
reversal rows with negative amounts and zero-distance rows. Each trip is assigned to exactly one
unit using its flow:

| Flow | Rule | Assigned to |
|---|---|---|
| `within` | Both ends in core | Pickup zone |
| `leaving` | Pickup in core, dropoff not | Pickup zone |
| `entering` | Dropoff in core, pickup not | Dropoff zone |
| `mixed` | No core end, at least one mixed end | The mixed zone |
| `outside` | Neither end in core or mixed | Pickup zone |

Core units count `entering + leaving + within`. Comparison units count `outside` trips only.
This matters: a comparison zone's raw pickups include trips headed into the zone, which are
treated. Including them would mix treated trips into the control. That naive version is kept as
a robustness check to show the difference.

Route is unobserved, so trips through the zone are visible only through the fee column.

Green cabs and non-high-volume FHV bases are out of scope.

## Comparison groups and the SUTVA problem

Drivers and riders displaced from the zone do not vanish. A rider who gets out at 61st Street, a
driver who waits in Long Island City, a commuter who now takes an Uber in Queens: all of these
change trip counts outside the zone. An outside-zone comparison therefore violates the
no-interference assumption, and the direction of the bias depends on the direction of the
displacement. This is treated as the central threat, by modelling exposure rather than
assuming it away, following the approach of Butts (2023) to spatial spillovers in
difference-in-differences:

| Group | Zones | Role |
|---|---|---|
| `core` | 37 | Treated |
| `mixed` | 7 | Partially charged. Own coefficient. Excluded from comparison |
| `surcharge_ring` | 236, 238, 239, 262, 263: Upper East and West Side, Yorkville | Exposed. Own coefficient. Same 2019 surcharge, no CRZ charge |
| `near` | Other zones under 4 km from the geofence | Exposed. Own coefficient |
| `far` | Zones at least 4 km from the geofence | Reference |

JFK, LaGuardia and Newark are never comparison units because air travel has its own demand path.
Zones 264 and 265 (unknown, outside NYC) are not units either. Trips between core zones and any
of these still count as core trips, because they are charged.

The primary coefficient assumes spillovers vanish beyond 4 km. Spillovers are sized four ways:

1. The `surcharge_ring` and `near` coefficients themselves, in percent and in absolute trips.
2. A finer gradient across 4-8, 8-12 and 12+ km, using 12+ km as the reference.
3. Re-estimating with each comparison set: all outside zones pooled (naive), ring only, 8+ km,
   12+ km. The spread is an empirical measure of comparison-group dependence.
4. An analytic bound for displacement the gradient cannot see. If a share `d` of the change in
   core trips reappears uniformly in the reference group, then the estimate is
   `tau * (1 + d * s)`, where `s` is core trips divided by reference trips in the pre-period.
   This is reported for d = 0.25, 0.5 and 1.

A uniform citywide shift is not detectable from a gradient. Item 4 exists because of that.

## Pre-period and post-period

Primary window: weeks -104 to 89, or 8 January 2023 to 26 September 2026.

- 2023 starts after the yellow fare rise of December 2022, and 104 weeks gives two full seasonal
  cycles before the program.
- 2022 is held out of the primary window because it still contains pandemic recovery. It is
  used for the placebo year and as a robustness window.
- Weeks 52 to 89 extend the post-period to September 2026, the last published month. The last
  partial week is dropped. 2025 and 2026 are also reported separately.
- The June 2024 pause and the November 2024 revival both fall inside the pre-period, and the
  revival left only seven weeks of notice. 2024 is the event-study reference year, so
  anticipation in late 2024 would pull the estimate towards zero. A robustness check that ends
  the pre-period on 2 November 2024 (week -9) tests for this.

## Primary specification

```
log(trips_it) = zone_i + week_t + group × season + group × Thanksgiving
              + beta_core Post_t Core_i
              + beta_mixed Post_t Mixed_i
              + beta_ring Post_t Ring_i
              + beta_near Post_t Near_i + e_it
```

Weighted least squares, weights equal to the zone's mean weekly trips in the pre-period, so the
coefficient approximates the proportional change in total trips rather than in the average
zone. `season` is the week number modulo 52: Christmas and New Year fall in the same seasonal
bin in every year of the sample, which was checked, and Thanksgiving, which does not, gets its
own term. Group by season terms are needed because the zone's seasonal shape (December tourism,
summer) differs from outer-borough seasonality, and plain two-way fixed effects would read that
as treatment.

With a single adoption date there is no staggered-timing problem. PPML is reported as a check
because weighted log OLS and PPML can differ when effects are heterogeneous across zones of
different sizes.

Units need a positive count in every week and at least 25 trips a week on average in the
pre-period.

## Inference

The core is one contiguous area treated once. Honest uncertainty is larger than a count of
trips suggests, and no single procedure settles it, so three are reported:

1. Cluster-robust standard errors at spatial clusters. Clusters are deterministic Ward clusters
   of zone centroids within group and borough, about six zones each, which gives about 6 core
   clusters.
2. Wild cluster restricted bootstrap (Webb six-point weights, 9,999 draws), with confidence
   intervals by test inversion. MacKinnon and Webb (2017) show this procedure under-rejects
   with few treated clusters, so it is not used alone.
3. Randomisation inference over placebo blocks: each far cluster is treated as pseudo-core and
   put through the same estimator. The question this answers is whether the core's change is
   larger than the changes other parts of the city showed anyway.

As a check, adjacent Ward clusters are merged pairwise, which halves the cluster count. Clusters
cannot simply go to borough level because the whole core would become one cluster, and a single
treated cluster has no within-treatment variation to estimate a variance from.

## Event study

Each 4-week bin is compared with the same weeks of 2024, the last full year before the program.
2024 is the reference year, so its bins are zero by construction. Bins in 2023 are a pre-trend
check. Bins in 2022 show pandemic recovery. Group by season terms are identified from 2024 alone,
which avoids collinearity between saturated event bins and seasonal terms.

The joint Wald pre-trend test is not trustworthy with six treated clusters because the covariance
matrix is rank deficient. The single-degree-of-freedom placebo test below takes its place, and
the largest pre-period bin is reported alongside it.

## Synthetic control

The treated unit is the aggregate core series. Donors are far clusters, aggregated to reduce noise.

- Classic synthetic control with an intercept and simplex weights, fit on log trips over the
  primary pre-window.
- Synthetic difference-in-differences (Arkhangelsky et al. 2021) with unit and time weights.
- A seasonally differenced variant (52-week log change) for 2025 only.
- Inference by in-space placebo: every donor takes a turn as treated. The post/pre RMSPE ratio
  rank gives a p-value and the placebo standard deviation gives an interval. Donors are smaller
  and noisier than the core, so this is conservative.
- In-time placebo with a fake start on 7 January 2024, fit on 2022 and 2023.

## Placebos

- Dates: a fake start in week -52 (7 January 2024) with the identical specification and the
  sample ending in week -1, then rolling fake starts every 13 weeks from -78 to -13.
- Zones: every far cluster as pseudo-treated, with the true core, mixed, ring and near zones
  removed.
- Dose: the `mixed` coefficient should be a fraction of the core coefficient roughly in line
  with the charged-trip share if the effect runs through the charge.
- Leakage: charged share of comparison flows.

## Pre-specified robustness set

By flow (entering, leaving, within); by service (yellow, high-volume); 2025 alone and 2026 alone;
peak and overnight; strict core of 20 fully contained zones; core plus mixed as treated; the
comparison sets above; naive zone pickups; unweighted log OLS; PPML; pre-window from 2022; no
seasonal terms; a core-specific linear pre-trend; pre-period ending before the November 2024
revival; holiday weeks dropped; the 2026 World Cup weeks (11 June to 19 July 2026) dropped.

## How the result will be described

The wording is fixed in advance.

- **Detected**: the primary wild bootstrap interval excludes zero, randomisation p is below 0.10,
  the fake-start placebo is smaller than the estimate and its wild interval includes zero,
  and the synthetic estimates share the sign.
- **Sensitive**: the primary estimate meets some of those tests but not all, or the sign changes
  across the comparison sets.
- **Null**: the interval includes zero. If so, the report gives the size of effect the interval
  rules out rather than calling it "no effect".

## What the data cannot show

- Effects on traffic, congestion, emissions, air quality or anyone not in a TLC trip record.
- Whether a lost trip moved to the subway, a bicycle, a private car, or disappeared.
- Routes. Passing through is visible only through the fee column.
- Drivers. Records have no driver identifier, so idle repositioning is invisible.
- Causal separation from anything else that hit the core alone in January 2025 and persisted,
  such as a return-to-office wave or a change in international tourism. Placebos test for such
  shocks in other periods and places. They cannot rule one out in this period and place.

## Deviations from this design

None yet.
