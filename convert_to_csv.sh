rm profile/*.csv
for REPORT in `ls profile/`; do
  # ncu --csv -i profile/$REPORT --metrics sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active > profile/$REPORT.csv
   ncu --csv -i profile/$REPORT > profile/$REPORT.csv
done