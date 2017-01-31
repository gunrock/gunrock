
# Comparison on Different GPUs

We ran Gunrock on several GPUs on 5 primitives times 9 datasets. As the compute and memory bandwidth capabilities of the GPUs increase, so does Gunrock's performance.

\htmlonly

  <!-- Container for the visualization gunrock_gpus -->
  <div id="vis_gunrock_gpus"></div>
  <script>
  var vlSpec = {
    "mark": "point", 
    "data": {
        "values": [
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_hollywood-2009_Thu Dec  1 095837 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:58:37", 
                "m_teps": 5742.517578125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_hollywood-2009_Thu Dec  1 103118 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:31:18", 
                "m_teps": 3069.005859375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_hollywood-2009_Tue Nov 29 095047 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:50:47", 
                "m_teps": 10637.6953125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_hollywood-2009_Tue Nov 29 200525 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:05:25", 
                "m_teps": 5266.68115234375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_hollywood-2009_Wed Nov 30 141331 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:13:31", 
                "m_teps": 2539.78271484375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_indochina-2004_Thu Dec  1 095841 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:58:41", 
                "m_teps": 7952.46142578125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_indochina-2004_Thu Dec  1 103123 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:31:23", 
                "m_teps": 5074.7373046875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_indochina-2004_Tue Nov 29 095055 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:50:55", 
                "m_teps": 12931.1435546875, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_indochina-2004_Tue Nov 29 200528 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:05:28", 
                "m_teps": 7781.12060546875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_indochina-2004_Wed Nov 30 141335 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:13:35", 
                "m_teps": 3752.5322265625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rgg_n24_0.000548_Thu Dec  1 100226 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:02:26", 
                "m_teps": 722.2317504882812, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rgg_n24_0.000548_Thu Dec  1 103529 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:35:29", 
                "m_teps": 510.25341796875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rgg_n24_0.000548_Tue Nov 29 095431 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:54:31", 
                "m_teps": 1021.9720458984375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rgg_n24_0.000548_Tue Nov 29 200924 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:09:24", 
                "m_teps": 698.2745361328125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rgg_n24_0.000548_Wed Nov 30 141759 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:17:59", 
                "m_teps": 379.7506408691406, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n22_e64_Thu Dec  1 095902 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:59:02", 
                "m_teps": 2329.981201171875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n22_e64_Thu Dec  1 103144 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:31:44", 
                "m_teps": 1304.5194091796875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n22_e64_Tue Nov 29 095114 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:51:14", 
                "m_teps": 5331.87744140625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n22_e64_Tue Nov 29 200553 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:05:53", 
                "m_teps": 1925.25830078125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n22_e64_Wed Nov 30 141402 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:14:02", 
                "m_teps": 1140.6607666015625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n23_e32_Thu Dec  1 095932 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:59:32", 
                "m_teps": 1746.9415283203125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n23_e32_Thu Dec  1 103219 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:32:19", 
                "m_teps": 1056.6055908203125, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n23_e32_Tue Nov 29 095141 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:51:41", 
                "m_teps": 3999.49169921875, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n23_e32_Tue Nov 29 200626 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:06:26", 
                "m_teps": 1414.226318359375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n23_e32_Wed Nov 30 141441 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:14:41", 
                "m_teps": 926.8665161132812, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n24_e16_Thu Dec  1 100007 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:00:07", 
                "m_teps": 1432.943115234375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n24_e16_Thu Dec  1 103257 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:32:57", 
                "m_teps": 904.2923583984375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n24_e16_Tue Nov 29 095212 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:52:12", 
                "m_teps": 3257.218505859375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n24_e16_Tue Nov 29 200704 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:07:04", 
                "m_teps": 1103.63330078125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_rmat_n24_e16_Wed Nov 30 141526 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:15:26", 
                "m_teps": 793.8890380859375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_road_usa_Thu Dec  1 100029 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:00:29", 
                "m_teps": 130.23220825195312, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_road_usa_Thu Dec  1 103323 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:33:23", 
                "m_teps": 99.56543731689453, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_road_usa_Tue Nov 29 095232 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:52:32", 
                "m_teps": 145.63819885253906, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_road_usa_Tue Nov 29 200728 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:07:28", 
                "m_teps": 110.34382629394531, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_road_usa_Wed Nov 30 141555 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:15:55", 
                "m_teps": 72.7790298461914, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-LiveJournal1_Thu Dec  1 095824 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:58:24", 
                "m_teps": 1674.7545166015625, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-LiveJournal1_Thu Dec  1 103103 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:31:03", 
                "m_teps": 1120.7100830078125, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-LiveJournal1_Tue Nov 29 095035 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:50:35", 
                "m_teps": 3672.78173828125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-LiveJournal1_Tue Nov 29 200513 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:05:13", 
                "m_teps": 1344.8056640625, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-LiveJournal1_Wed Nov 30 141315 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:13:15", 
                "m_teps": 966.1119995117188, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-orkut_Thu Dec  1 095830 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:58:30", 
                "m_teps": 1631.3426513671875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-orkut_Thu Dec  1 103109 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:31:09", 
                "m_teps": 1069.2659912109375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-orkut_Tue Nov 29 095041 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:50:41", 
                "m_teps": 3837.598876953125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-orkut_Tue Nov 29 200518 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:05:18", 
                "m_teps": 1291.48583984375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BC_soc-orkut_Wed Nov 30 141321 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:13:21", 
                "m_teps": 928.92529296875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_hollywood-2009_Thu Dec  1 095222 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:52:22", 
                "m_teps": 37059.7890625, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_hollywood-2009_Thu Dec  1 101844 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:18:44", 
                "m_teps": 19126.181640625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_hollywood-2009_Tue Nov 29 094505 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:45:05", 
                "m_teps": 50103.89453125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_hollywood-2009_Tue Nov 29 195802 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:58:02", 
                "m_teps": 31427.15625, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_hollywood-2009_Wed Nov 30 135855 2016.json\">JSON output</a>", 
                "time": "2016-11-30 13:58:55", 
                "m_teps": 13115.8466796875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_indochina-2004_Thu Dec  1 095229 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:52:29", 
                "m_teps": 7998.51318359375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_indochina-2004_Thu Dec  1 101858 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:18:58", 
                "m_teps": 3858.609619140625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_indochina-2004_Tue Nov 29 094511 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:45:11", 
                "m_teps": 11567.7177734375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_indochina-2004_Tue Nov 29 195817 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:58:17", 
                "m_teps": 7518.9345703125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_indochina-2004_Wed Nov 30 135915 2016.json\">JSON output</a>", 
                "time": "2016-11-30 13:59:15", 
                "m_teps": 2657.0712890625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rgg_n24_0.000548_Thu Dec  1 095403 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:54:03", 
                "m_teps": 730.2994384765625, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rgg_n24_0.000548_Thu Dec  1 102038 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:20:38", 
                "m_teps": 452.16717529296875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rgg_n24_0.000548_Tue Nov 29 094640 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:46:40", 
                "m_teps": 881.8350830078125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rgg_n24_0.000548_Tue Nov 29 200005 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:00:05", 
                "m_teps": 610.444091796875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rgg_n24_0.000548_Wed Nov 30 140116 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:01:16", 
                "m_teps": 363.31719970703125, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n22_e64_Thu Dec  1 095250 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:52:50", 
                "m_teps": 199440.46875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n22_e64_Thu Dec  1 101921 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:19:21", 
                "m_teps": 119147.390625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n22_e64_Tue Nov 29 094530 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:45:30", 
                "m_teps": 291780.625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n22_e64_Tue Nov 29 195843 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:58:43", 
                "m_teps": 183201.25, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n22_e64_Wed Nov 30 135942 2016.json\">JSON output</a>", 
                "time": "2016-11-30 13:59:42", 
                "m_teps": 86655.8359375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n23_e32_Thu Dec  1 095314 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:53:14", 
                "m_teps": 110766.078125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n23_e32_Thu Dec  1 101944 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:19:44", 
                "m_teps": 62196.6015625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n23_e32_Tue Nov 29 094553 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:45:53", 
                "m_teps": 165786.34375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n23_e32_Tue Nov 29 195909 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:59:09", 
                "m_teps": 101556.015625, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n23_e32_Wed Nov 30 140011 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:00:11", 
                "m_teps": 44445.1015625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n24_e16_Thu Dec  1 095338 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:53:38", 
                "m_teps": 56611.48046875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n24_e16_Thu Dec  1 102009 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:20:09", 
                "m_teps": 30624.197265625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n24_e16_Tue Nov 29 094616 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:46:16", 
                "m_teps": 88728.765625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n24_e16_Tue Nov 29 195935 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:59:35", 
                "m_teps": 51577.875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_rmat_n24_e16_Wed Nov 30 140041 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:00:41", 
                "m_teps": 21650.60546875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_road_usa_Thu Dec  1 095346 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:53:46", 
                "m_teps": 116.3349609375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_road_usa_Thu Dec  1 102019 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:20:19", 
                "m_teps": 89.5157699584961, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_road_usa_Tue Nov 29 094623 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:46:23", 
                "m_teps": 129.196533203125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_road_usa_Tue Nov 29 195946 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:59:46", 
                "m_teps": 101.86675262451172, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_road_usa_Wed Nov 30 140053 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:00:53", 
                "m_teps": 63.72743225097656, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-LiveJournal1_Thu Dec  1 095211 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:52:11", 
                "m_teps": 10524.9248046875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-LiveJournal1_Thu Dec  1 101826 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:18:26", 
                "m_teps": 6031.9921875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-LiveJournal1_Tue Nov 29 094456 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:44:56", 
                "m_teps": 18388.947265625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-LiveJournal1_Tue Nov 29 195743 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:57:43", 
                "m_teps": 10171.71875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-LiveJournal1_Wed Nov 30 135823 2016.json\">JSON output</a>", 
                "time": "2016-11-30 13:58:23", 
                "m_teps": 4236.5234375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-orkut_Thu Dec  1 095217 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:52:17", 
                "m_teps": 69273.796875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-orkut_Thu Dec  1 101837 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:18:37", 
                "m_teps": 37525.734375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-orkut_Tue Nov 29 094501 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:45:01", 
                "m_teps": 93399.2734375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-orkut_Tue Nov 29 195754 2016.json\">JSON output</a>", 
                "time": "2016-11-29 19:57:54", 
                "m_teps": 58253.41015625, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "BFS", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/BFS_soc-orkut_Wed Nov 30 135841 2016.json\">JSON output</a>", 
                "time": "2016-11-30 13:58:41", 
                "m_teps": 26812.587890625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_hollywood-2009_Thu Dec  1 101004 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:10:04", 
                "m_teps": 1497.9632568359375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_hollywood-2009_Thu Dec  1 104552 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:45:52", 
                "m_teps": 1195.7252197265625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_hollywood-2009_Tue Nov 29 095726 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:57:26", 
                "m_teps": 3056.11572265625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_hollywood-2009_Tue Nov 29 201205 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:12:05", 
                "m_teps": 1587.1085205078125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_hollywood-2009_Wed Nov 30 142105 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:21:05", 
                "m_teps": 808.2301635742188, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_indochina-2004_Thu Dec  1 101034 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:10:34", 
                "m_teps": 668.0885620117188, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_indochina-2004_Thu Dec  1 104621 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:46:21", 
                "m_teps": 338.67218017578125, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_indochina-2004_Tue Nov 29 095757 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:57:57", 
                "m_teps": 1496.7493896484375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_indochina-2004_Tue Nov 29 201229 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:12:29", 
                "m_teps": 557.4061889648438, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_indochina-2004_Wed Nov 30 142131 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:21:31", 
                "m_teps": 275.68157958984375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rgg_n24_0.000548_Thu Dec  1 102243 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:22:43", 
                "m_teps": 1357.5457763671875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rgg_n24_0.000548_Thu Dec  1 105823 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:58:23", 
                "m_teps": 727.2393188476562, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rgg_n24_0.000548_Tue Nov 29 100931 2016.json\">JSON output</a>", 
                "time": "2016-11-29 10:09:31", 
                "m_teps": 2251.191162109375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rgg_n24_0.000548_Tue Nov 29 202349 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:23:49", 
                "m_teps": 1377.255615234375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rgg_n24_0.000548_Wed Nov 30 143320 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:33:20", 
                "m_teps": 541.2537231445312, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n22_e64_Thu Dec  1 101150 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:11:50", 
                "m_teps": 1941.1944580078125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n22_e64_Thu Dec  1 104739 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:47:39", 
                "m_teps": 1109.45654296875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n22_e64_Tue Nov 29 095908 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:59:08", 
                "m_teps": 4152.36279296875, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n22_e64_Tue Nov 29 201343 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:13:43", 
                "m_teps": 1832.0345458984375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n22_e64_Wed Nov 30 142252 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:22:52", 
                "m_teps": 861.90966796875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n23_e32_Thu Dec  1 101508 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:15:08", 
                "m_teps": 1509.8446044921875, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n23_e32_Thu Dec  1 105045 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:50:45", 
                "m_teps": 927.3080444335938, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n23_e32_Tue Nov 29 100201 2016.json\">JSON output</a>", 
                "time": "2016-11-29 10:02:01", 
                "m_teps": 3266.63330078125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n23_e32_Tue Nov 29 201655 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:16:55", 
                "m_teps": 1298.04638671875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n23_e32_Wed Nov 30 142543 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:25:43", 
                "m_teps": 730.0156860351562, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n24_e16_Thu Dec  1 101832 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:18:32", 
                "m_teps": 1234.754150390625, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n24_e16_Thu Dec  1 105400 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:54:00", 
                "m_teps": 733.84130859375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n24_e16_Tue Nov 29 100517 2016.json\">JSON output</a>", 
                "time": "2016-11-29 10:05:17", 
                "m_teps": 2570.96826171875, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n24_e16_Tue Nov 29 202005 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:20:05", 
                "m_teps": 1086.8001708984375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_rmat_n24_e16_Wed Nov 30 142859 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:28:59", 
                "m_teps": 619.019775390625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_road_usa_Thu Dec  1 102203 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:22:03", 
                "m_teps": 446.7572021484375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_road_usa_Thu Dec  1 105741 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:57:41", 
                "m_teps": 273.19866943359375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_road_usa_Tue Nov 29 100851 2016.json\">JSON output</a>", 
                "time": "2016-11-29 10:08:51", 
                "m_teps": 778.8450317382812, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_road_usa_Tue Nov 29 202311 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:23:11", 
                "m_teps": 451.9570007324219, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_road_usa_Wed Nov 30 143239 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:32:39", 
                "m_teps": 178.10897827148438, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-LiveJournal1_Thu Dec  1 100802 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:08:02", 
                "m_teps": 1389.5433349609375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-LiveJournal1_Thu Dec  1 104353 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:43:53", 
                "m_teps": 913.656005859375, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-LiveJournal1_Tue Nov 29 095523 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:55:23", 
                "m_teps": 2709.746337890625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-LiveJournal1_Tue Nov 29 201017 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:10:17", 
                "m_teps": 1296.412353515625, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-LiveJournal1_Wed Nov 30 141907 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:19:07", 
                "m_teps": 696.8785400390625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-orkut_Thu Dec  1 100839 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:08:39", 
                "m_teps": 1901.1956787109375, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-orkut_Thu Dec  1 104429 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:44:29", 
                "m_teps": 971.9141235351562, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-orkut_Tue Nov 29 095602 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:56:02", 
                "m_teps": 3875.7724609375, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-orkut_Tue Nov 29 201049 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:10:49", 
                "m_teps": 1614.6390380859375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "CC", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/CC_soc-orkut_Wed Nov 30 141945 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:19:45", 
                "m_teps": 820.7590942382812, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_hollywood-2009_Thu Dec  1 100401 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:04:01", 
                "m_teps": 8976.655517578125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_hollywood-2009_Thu Dec  1 103730 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:37:30", 
                "m_teps": 5558.8839111328125, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_hollywood-2009_Wed Nov 30 143652 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:36:52", 
                "m_teps": 7446.57421875, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_hollywood-2009_Wed Nov 30 144258 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:42:58", 
                "m_teps": 6049.0594482421875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_hollywood-2009_Wed Nov 30 175141 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:51:41", 
                "m_teps": 16949.92822265625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_indochina-2004_Thu Dec  1 100406 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:04:06", 
                "m_teps": 9996.382690429688, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_indochina-2004_Thu Dec  1 103737 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:37:37", 
                "m_teps": 7003.584167480469, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_indochina-2004_Wed Nov 30 143722 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:37:22", 
                "m_teps": 8275.868713378906, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_indochina-2004_Wed Nov 30 144307 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:43:07", 
                "m_teps": 7700.560791015625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_indochina-2004_Wed Nov 30 175147 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:51:47", 
                "m_teps": 20185.80712890625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rgg_n24_0.000548_Thu Dec  1 100740 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:07:40", 
                "m_teps": 2582.404586791992, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rgg_n24_0.000548_Thu Dec  1 104315 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:43:15", 
                "m_teps": 1435.7117080688477, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rgg_n24_0.000548_Wed Nov 30 144141 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:41:41", 
                "m_teps": 2322.553871154785, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rgg_n24_0.000548_Wed Nov 30 144810 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:48:10", 
                "m_teps": 1903.9076461791992, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rgg_n24_0.000548_Wed Nov 30 175347 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:53:47", 
                "m_teps": 4749.027908325195, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n22_e64_Thu Dec  1 100433 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:04:33", 
                "m_teps": 3364.303176879883, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n22_e64_Thu Dec  1 103808 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:38:08", 
                "m_teps": 1598.1565475463867, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n22_e64_Wed Nov 30 143754 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:37:54", 
                "m_teps": 2877.3841094970703, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n22_e64_Wed Nov 30 144337 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:43:37", 
                "m_teps": 1930.9824142456055, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n22_e64_Wed Nov 30 175208 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:52:08", 
                "m_teps": 8754.96957397461, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n23_e32_Thu Dec  1 100518 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:05:18", 
                "m_teps": 2323.3816528320312, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n23_e32_Thu Dec  1 103925 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:39:25", 
                "m_teps": 1254.887580871582, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n23_e32_Wed Nov 30 143850 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:38:50", 
                "m_teps": 1941.0651473999023, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n23_e32_Wed Nov 30 144444 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:44:44", 
                "m_teps": 1517.1942138671875, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n23_e32_Wed Nov 30 175236 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:52:36", 
                "m_teps": 6616.79638671875, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n24_e16_Thu Dec  1 100621 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:06:21", 
                "m_teps": 1765.9974670410156, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n24_e16_Thu Dec  1 104101 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:41:01", 
                "m_teps": 1044.2986297607422, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n24_e16_Wed Nov 30 144003 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:40:03", 
                "m_teps": 1455.7415771484375, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n24_e16_Wed Nov 30 144611 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:46:11", 
                "m_teps": 1261.2461853027344, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_rmat_n24_e16_Wed Nov 30 175308 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:53:08", 
                "m_teps": 4626.863098144531, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_road_usa_Thu Dec  1 100722 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:07:22", 
                "m_teps": 1805.5992965698242, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_road_usa_Thu Dec  1 104243 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:42:43", 
                "m_teps": 596.4975643157959, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_road_usa_Wed Nov 30 144121 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:41:21", 
                "m_teps": 1530.5845642089844, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_road_usa_Wed Nov 30 144738 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:47:38", 
                "m_teps": 598.5783061981201, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_road_usa_Wed Nov 30 175333 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:53:33", 
                "m_teps": 4594.353057861328, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-LiveJournal1_Thu Dec  1 100323 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:03:23", 
                "m_teps": 2398.4324340820312, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-LiveJournal1_Thu Dec  1 103633 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:36:33", 
                "m_teps": 1556.6260986328125, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-LiveJournal1_Wed Nov 30 143544 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:35:44", 
                "m_teps": 2001.0831298828125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-LiveJournal1_Wed Nov 30 144209 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:42:09", 
                "m_teps": 1923.3012084960938, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-LiveJournal1_Wed Nov 30 175122 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:51:22", 
                "m_teps": 6475.663330078125, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-orkut_Thu Dec  1 100334 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:03:34", 
                "m_teps": 1915.1980895996094, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-orkut_Thu Dec  1 103649 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:36:49", 
                "m_teps": 1224.7851943969727, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-orkut_Wed Nov 30 143613 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:36:13", 
                "m_teps": 1572.275405883789, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-orkut_Wed Nov 30 144223 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:42:23", 
                "m_teps": 1496.3440856933594, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "PageRank", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/PageRank_soc-orkut_Wed Nov 30 175129 2016.json\">JSON output</a>", 
                "time": "2016-11-30 17:51:29", 
                "m_teps": 5407.619201660156, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_hollywood-2009_Thu Dec  1 102218 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:22:18", 
                "m_teps": 1448.8941650390625, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "hollywood-2009", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_hollywood-2009_Wed Nov 30 140328 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:03:28", 
                "m_teps": 1000.3242797851562, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_indochina-2004_Thu Dec  1 102255 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:22:55", 
                "m_teps": 670.0359497070312, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "indochina-2004", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_indochina-2004_Wed Nov 30 140428 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:04:28", 
                "m_teps": 473.1410217285156, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rgg_n24_0.000548_Thu Dec  1 095728 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:57:28", 
                "m_teps": 772.6026611328125, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rgg_n24_0.000548_Thu Dec  1 103007 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:30:07", 
                "m_teps": 403.90350341796875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rgg_n24_0.000548_Tue Nov 29 094937 2016.json\">JSON output</a>", 
                "time": "2016-11-29 09:49:37", 
                "m_teps": 754.1578369140625, 
                "gpuinfo.name": "Tesla P100-PCIE-16GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rgg_n24_0.000548_Tue Nov 29 200422 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:04:22", 
                "m_teps": 582.5126953125, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rgg_n24_0.000548", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rgg_n24_0.000548_Wed Nov 30 141218 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:12:18", 
                "m_teps": 270.98675537109375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n22_e64_Thu Dec  1 102356 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:23:56", 
                "m_teps": 828.6013793945312, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n22_e64", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n22_e64_Wed Nov 30 140533 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:05:33", 
                "m_teps": 689.7913818359375, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n23_e32_Thu Dec  1 102513 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:25:13", 
                "m_teps": 671.43310546875, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n23_e32", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n23_e32_Wed Nov 30 140701 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:07:01", 
                "m_teps": 579.7408447265625, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n24_e16_Thu Dec  1 102639 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:26:39", 
                "m_teps": 572.5731811523438, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "rmat_n24_e16", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_rmat_n24_e16_Wed Nov 30 140830 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:08:30", 
                "m_teps": 496.9857482910156, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_road_usa_Thu Dec  1 095559 2016.json\">JSON output</a>", 
                "time": "2016-12-01 09:55:59", 
                "m_teps": 9.88219165802002, 
                "gpuinfo.name": "Tesla M40 24GB", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_road_usa_Thu Dec  1 102801 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:28:01", 
                "m_teps": 6.071076393127441, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_road_usa_Tue Nov 29 200240 2016.json\">JSON output</a>", 
                "time": "2016-11-29 20:02:40", 
                "m_teps": 8.146416664123535, 
                "gpuinfo.name": "Tesla M40", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "road_usa", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_road_usa_Wed Nov 30 140954 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:09:54", 
                "m_teps": 5.1867241859436035, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_soc-LiveJournal1_Thu Dec  1 102055 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:20:55", 
                "m_teps": 203.72372436523438, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "soc-LiveJournal1", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_soc-LiveJournal1_Wed Nov 30 140139 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:01:39", 
                "m_teps": 181.11741638183594, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_soc-orkut_Thu Dec  1 102131 2016.json\">JSON output</a>", 
                "time": "2016-12-01 10:21:31", 
                "m_teps": 253.3397674560547, 
                "gpuinfo.name": "Tesla K40m", 
                "gunrock_version": "0.4.0"
            }, 
            {
                "engine": "Gunrock", 
                "algorithm": "SSSP", 
                "dataset": "soc-orkut", 
                "details": "<a href=\"https://github.com/gunrock/io/tree/master/gunrock-output/topc/CentOS7.2_XXx1_topc_arch/SSSP_soc-orkut_Wed Nov 30 140232 2016.json\">JSON output</a>", 
                "time": "2016-11-30 14:02:32", 
                "m_teps": 222.9230194091797, 
                "gpuinfo.name": "Tesla K80", 
                "gunrock_version": "0.4.0"
            }
        ]
    }, 
    "encoding": {
        "y": {
            "field": "m_teps", 
            "scale": {
                "type": "log"
            }, 
            "type": "quantitative", 
            "axis": {
                "title": "MTEPS"
            }
        }, 
        "color": {
            "field": "[gpuinfo.name]", 
            "type": "nominal", 
            "legend": {
                "title": "GPU"
            }
        }, 
        "shape": {
            "field": "[gpuinfo.name]", 
            "type": "nominal", 
            "legend": {
                "title": "GPU"
            }
        }, 
        "column": {
            "field": "algorithm", 
            "type": "nominal", 
            "axis": {
                "orient": "top", 
                "title": "Primitive"
            }
        }, 
        "x": {
            "field": "dataset", 
            "type": "nominal", 
            "axis": {
                "title": "Dataset"
            }
        }
    }
}
  var embedSpec = {
    mode: "vega-lite",  // Instruct Vega-Embed to use the Vega-Lite compiler
    spec: vlSpec
  };
  // Embed the visualization in the container with id `vis_gunrock_gpus`
  vg.embed("#vis_gunrock_gpus", embedSpec, function(error, result) {
    // Callback receiving the View instance and parsed Vega spec
    // result.view is the View, which resides under the
    // '#vis_gunrock_gpus' element
  });
  </script>

\endhtmlonly


[Source data](md_stats_gunrock_gpus_table_html.html), with links to the output JSON for each run
