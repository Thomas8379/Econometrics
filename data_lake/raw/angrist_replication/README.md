# angrist_replication

Replication datasets from Joshua Angrist's Harvard Dataverse deposits.

## Contents

| Filename | Paper | Journal | DOI |
|----------|-------|---------|-----|
| `asciiqob.tab` | Angrist & Krueger (1991) — Compulsory schooling, quarter-of-birth IV | QJE | [10.7910/DVN/ENLGZX](https://doi.org/10.7910/DVN/ENLGZX) |
| `asciiqob.zip` | Angrist & Krueger (1991) — original zip archive | QJE | [10.7910/DVN/ENLGZX](https://doi.org/10.7910/DVN/ENLGZX) |
| `ak91.sas` | Angrist & Krueger (1991) — SAS analysis code | QJE | [10.7910/DVN/ENLGZX](https://doi.org/10.7910/DVN/ENLGZX) |
| `cwhsa.tab`, `cwhsb.tab`, `cwhsc_new.tab` | Angrist (1990) — CWHS earnings records, Vietnam draft lottery | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `sipp2.tab` | Angrist (1990) — SIPP survey, veteran effects | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `dmdcdat.tab` | Angrist (1990) — DMDC military records | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `Draft_Lottery_Numbers-lott_69.csv` … `lott_72.csv` | Angrist (1990) — Vietnam-era draft lottery numbers 1969–1972 | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `Angrist1990_Table*.do` | Angrist (1990) — Stata replication do-files | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `Data.zip` | Angrist (1990) — original zip archive | AER | [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL) |
| `final4.tab`, `final5.tab` | Angrist & Lavy (1999) — Israeli class-size data, grades 4 & 5 | QJE | [10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU) |
| `AngristLavy_Table*.do` | Angrist & Lavy (1999) — Stata replication do-files | QJE | [10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU) |
| `mmoulton_post.do` | Angrist & Lavy (1999) — Moulton correction code | QJE | [10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU) |
| `data__1232173.zip` | Angrist & Lavy (1999) — original zip (renamed to avoid collision with `Data.zip`) | QJE | [10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU) |
| `Angrist1993_Table*.do` | Angrist (1993) — Stata replication do-files | JHR | [10.7910/DVN/CAQYME](https://doi.org/10.7910/DVN/CAQYME) |
| `soviii_ang93b.tab`, `soviii_ang93b.zip` | Angrist (1993) — SOVI-VIII survey data (veterans benefits/education) | JHR | [10.7910/DVN/CAQYME](https://doi.org/10.7910/DVN/CAQYME) |
| `aerdat4.tab` | Angrist et al. (2002) — Colombia PACES voucher data (main survey dataset) | AER | [10.7910/DVN/K57TOZ](https://doi.org/10.7910/DVN/K57TOZ) |
| `tab5v1.tab`, `tab7.tab`, `tab7test.tab` | Angrist et al. (2002) — Colombia supplementary tables | AER | [10.7910/DVN/K57TOZ](https://doi.org/10.7910/DVN/K57TOZ) |
| `table*_final.sas` | Angrist et al. (2002) — SAS replication code | AER | [10.7910/DVN/K57TOZ](https://doi.org/10.7910/DVN/K57TOZ) |
| `data__1232807.zip`, `file1.txt`, `ReadPolice1b.sas` | Angrist & Krueger (2001) — IV methods in experimental settings (police data) | JEP | [10.7910/DVN/FTZ8GN](https://doi.org/10.7910/DVN/FTZ8GN) |

## Papers

### Angrist (1993) — Veterans Benefits and Schooling
*The Effect of Veterans Benefits on Education and Earnings*
Journal of Human Resources. DOI: [10.7910/DVN/CAQYME](https://doi.org/10.7910/DVN/CAQYME)

`soviii_ang93b.tab` is the Survey of Veterans (SOVI-VIII) dataset used to estimate
how GI Bill and VA educational benefits affect schooling and earnings among Vietnam-era veterans.
Key comparison: Vietnam veterans vs. non-Vietnam-era AVF veterans.

### Angrist, Bettinger, Bloom, King & Kremer (2002) — Colombia PACES Vouchers
*Vouchers for Private Schooling in Colombia: Evidence from a Randomised Natural Experiment*
American Economic Review. DOI: [10.7910/DVN/K57TOZ](https://doi.org/10.7910/DVN/K57TOZ)

`aerdat4.tab` contains survey follow-up data on applicants to the PACES voucher lottery.
The lottery (vouch0) is a clean randomised instrument for private school attendance.
Key results: voucher winners completed more schooling (scyfnsh ≈ +0.10 years) and were
more likely to attend private schools (prscha_1 effect ≈ +0.06).

### Angrist & Krueger (1991) — Quarter-of-birth IV
*Does Compulsory School Attendance Affect Schooling and Earnings?*
Quarterly Journal of Economics. DOI: [10.7910/DVN/ENLGZX](https://doi.org/10.7910/DVN/ENLGZX)

`asciiqob.tab` is the primary dataset: ~329,000 men from the 1980 Census with quarter-of-birth
indicators used as instruments for years of schooling. This is the canonical example for
weak-instrument and JIVE analysis.

### Angrist (1990) — Vietnam Draft Lottery IV
*Lifetime Earnings and the Vietnam Era Draft Lottery: Evidence from Social Security Administrative Records*
American Economic Review. DOI: [10.7910/DVN/PLF0YL](https://doi.org/10.7910/DVN/PLF0YL)

Draft-lottery numbers as instruments for veteran status. Datasets cover CWHS earnings records,
SIPP surveys, and DMDC military records. The `Draft_Lottery_Numbers-lott_*.csv` files contain
the actual lottery draws for years 1969–1972.

### Angrist & Lavy (1999) — Maimonides Rule
*Using Maimonides' Rule to Estimate the Effect of Class Size on Scholastic Achievement*
Quarterly Journal of Economics. DOI: [10.7910/DVN/XRSUJU](https://doi.org/10.7910/DVN/XRSUJU)

Israeli school data with Maimonides' rule (class-size cap of 40) as an RD/IV instrument.
`final4.tab` and `final5.tab` are 4th- and 5th-grade test score datasets.

### Angrist & Krueger (2001) — IV Methods in Experimental Settings
*Instrumental Variables and the Search for Identification: From Supply and Demand to Natural Experiments*
Journal of Economic Perspectives. DOI: [10.7910/DVN/FTZ8GN](https://doi.org/10.7910/DVN/FTZ8GN)

Small police dataset used as a methodological illustration of IV in an experimental context.
`data__1232807.zip` contains the data; `ReadPolice1b.sas` contains the analysis code.

## Acquisition

Downloaded via `scripts/dataverse_fetch.py`. To re-fetch any deposit:

```bash
python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/ENLGZX \
    --dest data_lake/raw/angrist_replication

python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/PLF0YL \
    --dest data_lake/raw/angrist_replication

python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/XRSUJU \
    --dest data_lake/raw/angrist_replication

python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/CAQYME \
    --dest data_lake/raw/angrist_replication \
    --types tab zip do txt

python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/K57TOZ \
    --dest data_lake/raw/angrist_replication \
    --types tab sas txt

python scripts/dataverse_fetch.py fetch \
    --doi doi:10.7910/DVN/FTZ8GN \
    --dest data_lake/raw/angrist_replication \
    --types zip txt sas
```

Re-running is safe: files already present with matching hashes are skipped.

To search for additional Angrist deposits:

```bash
python scripts/dataverse_fetch.py search --query "Angrist" --max-results 20
```

## Licence

These datasets are deposited under the terms specified by each Dataverse entry.
Check the DOI landing page for the specific licence before redistribution.
