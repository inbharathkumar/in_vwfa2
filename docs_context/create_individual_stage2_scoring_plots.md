Read the instructions from the attached file. 
Go to Repo: https://github.com/inbharathkumar/in_vwfa2.git
Read the thesis_full_draft_no_cover.docx from /docs_context
Then read the Combinations List.md from /docs_context

The main code for the project is in "local_code_full.py". But, it doesn't have the code to generate Stage 2 ROI plots for individual scoring factors. I have provided a list of required plots in combinations_list.md file, and an updated plotting code is present in "code/Colab_Code-To_Save_ROI_PNG_Plots-New-With_Crop_Fix.py". Read below for what has to be added and what has to be removed/commented.


What to add in the new code?
Can you update the plotting code of the local_code_full.py with the new code from "code/Colab_Code-To_Save_ROI_PNG_Plots-New-With_Crop_Fix.py", and then use that code to generate the Stage 2 ROIs of individual scoring factors? 
So, you must be taking the ROI labels that was created at the end of Stage 1, and then apply individual scoring factors and then generate the Stage 2 ROI plots separately and save them in results folder (for this task no need to generate html outputs).
If you have any questions, ask me. Before you do anything, check if you have permission to install the packages, I have already provided internet access and installation access. I guess one of the tasks already involved installing those stuff, so you must be good to go!

What to ignore?
No need of generating both, the scaling plots and scaling ROI plots. Generate only Stage 1 ROIs.

Final Expected Output: A zip file of the results directory containing the below files.
Just to be clear, code that stitches all subject plots is already available the new plotting code.
results/<run_name>/0_stage1_roi-all.png 
results/<run_name>/0_stage1_roi-sub-01.png
results/<run_name>/0_stage1_roi-sub-02.png
results/<run_name>/0_stage1_roi-sub-03.png
results/<run_name>/0_stage1_roi-sub-04.png
results/<run_name>/0_stage1_roi-sub-05.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-all.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-sub-01.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-sub-02.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-sub-03.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-sub-04.png
results/<run_name>/1_stage2_roi_with_cortical_surface_scoring-sub-05.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-all.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-sub-01.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-sub-02.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-sub-03.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-sub-04.png
results/<run_name>/2_stage2_roi_with_parcellation_scoring-sub-05.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-all.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-sub-01.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-sub-02.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-sub-03.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-sub-04.png
results/<run_name>/3_stage2_roi_with_nontarget_contrast_scoring-sub-05.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-all.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-sub-01.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-sub-02.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-sub-03.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-sub-04.png
results/<run_name>/4_stage2_roi_with_session_count_scoring-sub-05.png