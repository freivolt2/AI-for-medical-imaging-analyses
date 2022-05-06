% This script runs a skull-stripping process on inputted images. (SPM12 needed)

clear all;

GM_DIRECTORY = 'E:\testSegment\NC\c1'; % gray matter directory
WM_DIRECTORY = 'E:\testSegment\NC\c2'; % white matter directory
O_DIRECTORY = 'E:\testSegment\NC\original'; % original image directory

OUT_DIRECTORY = 'E:\testSegment\NC\skull-stripped'; % output directory

gm_dir = dir(GM_DIRECTORY);
wm_dir = dir(WM_DIRECTORY);
o_dir = dir(O_DIRECTORY);

gm_files = {gm_dir.name};
wm_files = {wm_dir.name};
o_files = {o_dir.name};

gm_size = size(gm_files, 2);
wm_size = size(wm_files, 2);
o_size = size(o_files, 2);

size_match_condition = (eq(gm_size, wm_size) && eq(wm_size, o_size));

assert(size_match_condition, 'Number of gray matter, white matter and original files must me the same!');

for i = 3:o_size
    gm_file_name = gm_files(i);
    wm_file_name = wm_files(i);
    o_file_name = o_files(i);
    
    gm_full_dir = string(append(GM_DIRECTORY, '\', gm_file_name, ',1'));
    wm_full_dir = string(append(WM_DIRECTORY, '\', wm_file_name, ',1'));
    o_full_dir = string(append(O_DIRECTORY, '\', o_file_name, ',1'));
    
    new_file_name = char(string(append('s', erase(o_file_name, '.nii'))));
    
    matlabbatch{1}.spm.util.imcalc.input = cellstr([gm_full_dir wm_full_dir o_full_dir].');
    matlabbatch{1}.spm.util.imcalc.output = new_file_name;
    matlabbatch{1}.spm.util.imcalc.outdir = {OUT_DIRECTORY};
    matlabbatch{1}.spm.util.imcalc.expression = '(i3.*(i1+i2))';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
       
    %% Run Matlabbatch
    try
        spm('defaults', 'FMRI');
        spm_jobman('run',matlabbatch);
    catch
        errorMsg = 'Error running matlabbatch';
        disp(append(errorMsg, ' for ', new_file_name));
        return
    end
    disp(append(string(i-2), ' out of ', string(o_size-2)));
end