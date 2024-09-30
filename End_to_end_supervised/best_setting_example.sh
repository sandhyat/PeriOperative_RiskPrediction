
# temporary file

#!/usr/bin/env bash

python XGBT_tabular.py --task='mortality' --bestModel=True
python XGBT_tabular.py --homemeds --task='mortality' --bestModel=True
python XGBT_tabular.py --homemeds --pmhProblist --task='mortality' --bestModel=True

python XGBT_tabular.py --task='postop_los' --bestModel=True
python XGBT_tabular.py --homemeds --task='postop_los' --bestModel=True
python XGBT_tabular.py --homemeds --pmhProblist --task='postop_los' --bestModel=True

python XGBT_tabular.py --task='opioids_count_day0' --bestModel=True
python XGBT_tabular.py --homemeds --task='opioids_count_day0' --bestModel=True
python XGBT_tabular.py --homemeds --pmhProblist --task='opioids_count_day0' --bestModel=True

python XGBT_tabular.py --task='opioids_count_day1' --bestModel=True
python XGBT_tabular.py --homemeds --task='opioids_count_day1' --bestModel=True
python XGBT_tabular.py --homemeds --pmhProblist --task='opioids_count_day1' --bestModel=True

python Tabnet_tabular.py --task='aki2' --bestModel=True
python Tabnet_tabular.py --homemeds --task='aki2' --bestModel=True
python Tabnet_tabular.py --homemeds --pmhProblist --task='aki2' --bestModel=True

python Tabnet_tabular.py --task='mortality' --bestModel=True
python Tabnet_tabular.py --homemeds --task='mortality' --bestModel=True
python Tabnet_tabular.py --homemeds --pmhProblist --task='mortality' --bestModel=True

python Tabnet_tabular.py --task='opioids_count_day1' --bestModel=True
python Tabnet_tabular.py --homemeds --task='opioids_count_day1' --bestModel=True
python Tabnet_tabular.py --homemeds --pmhProblist --task='opioids_count_day1' --bestModel=True

python /codes/Two_stage_selfsupervised/Scarf_tabular.py --task='postop_los' --bestModel=True
python /codes/Two_stage_selfsupervised/Scarf_tabular.py --homemeds --task='postop_los' --bestModel=True
python /codes/Two_stage_selfsupervised/Scarf_tabular.py --homemeds --pmhProblist --task='opioids_count_day1' --bestModel=True

python /codes/Two_stage_selfsupervised/Scarf_tabular.py --task='mortality' --bestModel=True
python /codes/Two_stage_selfsupervised/Scarf_tabular.py --homemeds --task='mortality' --bestModel=True
python /codes/Two_stage_selfsupervised/Scarf_tabular.py --homemeds --pmhProblist --task='mortality' --bestModel=True

#-------------------------------------------------

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='icu' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='icu' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='icu' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='icu' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='icu' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='icu' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='mortality' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='mortality' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='mortality' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='mortality' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='mortality' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='mortality' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='opioids_count_day0' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='opioids_count_day0' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='opioids_count_day0' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='opioids_count_day0' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='opioids_count_day0' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day0' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='opioids_count_day1' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='opioids_count_day1' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='opioids_count_day1' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='opioids_count_day1' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='opioids_count_day1' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day1' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='aki2' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='aki2' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='aki2' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='aki2' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='aki2' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='aki2' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --task='postop_los' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --flow --task='postop_los' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --task='postop_los' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --task='postop_los' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --task='postop_los' --bestModel=True
python /codes/End_to_end_supervised/Training_with_TS_Summary.py --meds --flow --preops --homemeds --pmhProblist --task='postop_los' --bestModel=True

python /codes/End_to_end_supervised/Training_with_TimeSeries.py --meds --task='mortality' --bestModel=True --epochs=10
python /codes/End_to_end_supervised/Training_with_TimeSeries.py --flow --task='mortality' --bestModel=True --epochs=10
python /codes/End_to_end_supervised/Training_with_TimeSeries.py --meds --flow --task='mortality' --bestModel=True --epochs=10
python /codes/End_to_end_supervised/Training_with_TimeSeries.py --meds --flow --preops --task='mortality' --bestModel=True --epochs=10
python /codes/End_to_end_supervised/Training_with_TimeSeries.py --meds --flow --preops --homemeds --task='mortality' --bestModel=True --epochs=10
python /codes/End_to_end_supervised/Training_with_TimeSeries.py --meds --flow --preops --homemeds --pmhProblist --task='mortality' --bestModel=True --epochs=10

#----------------------------------------------------------- Wave 2-------------------------------------------------

# temporal validation of xgbtts summary models

python /codes/TS_wave2.py  --meds --task='postop_los'
python /codes/TS_wave2.py  --flow --task='postop_los'
python /codes/TS_wave2.py  --meds --flow --task='postop_los'
python /codes/TS_wave2.py  --meds --flow --preops --task='postop_los'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --task='postop_los'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --pmhProblist --task='postop_los'

python /codes/TS_wave2.py  --meds --task='opioids_count_day0'
python /codes/TS_wave2.py  --flow --task='opioids_count_day0'
python /codes/TS_wave2.py  --meds --flow --task='opioids_count_day0'
python /codes/TS_wave2.py  --meds --flow --preops --task='opioids_count_day0'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --task='opioids_count_day0'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day0'

python /codes/TS_wave2.py  --meds --task='opioids_count_day1'
python /codes/TS_wave2.py  --flow --task='opioids_count_day1'
python /codes/TS_wave2.py  --meds --flow --task='opioids_count_day1'
python /codes/TS_wave2.py  --meds --flow --preops --task='opioids_count_day1'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --task='opioids_count_day1'
python /codes/TS_wave2.py  --meds --flow --preops --homemeds --pmhProblist --task='opioids_count_day1'