%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script properties
% name : IVIMfit
% Description : performs the IVIM fit with unregularized NNLS
% Arguments :
%   Input:
%       exploreDTI_file     :   ExploreDTI MAT-file with the preprocessed IVIM data
%       bval                :   Text file with the b-values
%       mask                :   Brain mask (must be in the same space/orientation)
%       output_folder       :   Specify the name of the output folder
%       output_name         :   Specify the name of the ouput files
%       TE                  :   echo time of the IVIM scan
%       TI                  :   inversion time of the IVIM scan (set to -1 if no inversion pulse is applied)
%       TR                  :   repetition time of the IVIM scan
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Please cite the following paper when using this code:
% Wong et al. Spectral Diffusion Analysis of Intravoxel Incoherent Motion 
% MRI in Cerebral Small Vessel Disease. 
% doi: https://doi.org/10.1002/jmri.26920
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version History
% 20221102 1.0 Gerald Drenthen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function IVIMfit(exploreDTI_file, bval, mask, output_folder, output_name, TE, TI, TR)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Relaxation times of the three compartments
T1blood = 1624e-3;
T2blood = 275e-3; 
T1tissue = 1081e-3; 
T2tissue = 95e-3; 
T1pvs = 1250e-3;
T2pvs = 500e-3;

bval = load(bval);
nbval = length(bval);

Dlength = 80;                       % Number of basis functions
Dmin = 0.1e-3;                      % Lowest diffusivity
Dmax = 200e-3;                      % Highest diffusivity
Dspace = logspace( log10( Dmin ), log10( Dmax ), Dlength );
DBasis = exp(-kron(bval',Dspace));  % Set of basis functions
Dpvs_min = 1.5e-3;
Dpvs_max = 4e-3;
par_range = (Dspace<Dpvs_min);
pvs_range = (Dspace<Dpvs_max)-par_range;
perf_range = Dspace>Dpvs_max; 

if TI>0                             % If inversion pulse is applied
    E1pvs = abs((1-2*exp(-TI/T1pvs) + exp(-TR/T1pvs)));
    E1perf = (1-exp(-TR/T1blood));
    E1par = (1-2*exp(-TI/T1tissue) + exp(-TR/T1tissue));
else                                % Else, no inversion pulse
    E1pvs = (1-exp(-TR/T1pvs));
    E1perf = (1-exp(-TR/T1blood));
    E1par = (1-exp(-TR/T1tissue));
end
E2pvs = exp(-(TE/T2pvs));
E2perf = exp(-(TE/T2blood));
E2par = exp(-(TE/T2tissue));
                    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load data and filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(exploreDTI_file,'DWI')

DWI_mat = double(cat(4,DWI{:}));

nii = load_untouch_nii(mask);
brain_mask = permute(nii.img(end:-1:1,end:-1:1,:),[2 1 3]) > 0;

X = size(DWI_mat,1);
Y = size(DWI_mat,2);
Z = size(DWI_mat,3);

% optional, svd filter (refer to: 10.1016/j.mri.2006.03.006)
DWI_filt = reshape(SVD_filter(reshape(DWI_mat,[X Y*Z nbval])),[X Y Z nbval]);

% optional, guassian filter:
H = fspecial3('gaussian',[3 3 3],2/2.355); 
for bvaln = 1:size(DWI_mat,4)
    DWI_filt(:,:,:,bvaln) = imfilter(DWI_filt(:,:,:,bvaln),H) .* brain_mask;
end

D_par = zeros(X,Y,Z);
D_pvs = zeros(X,Y,Z);
D_perf = zeros(X,Y,Z);
f_pvsCorr = zeros(X,Y,Z);
f_blCorr = zeros(X,Y,Z);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the IVIM fit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning('off') % surpress warnings

for SL = 1:Z
    disp(['current slice: ' num2str(SL) '/' num2str(size(DWI_mat,3))])
    for YY = 1:Y    
        for XX = 1:X
            decay = squeeze(DWI_filt(XX,YY,SL,:));
            if  brain_mask(XX,YY,SL) > 0
                amplitudes = lsqnonneg(DBasis, decay);

                f_pvs = sum(amplitudes(pvs_range>0))/sum(amplitudes(:));
                f_perf = sum(amplitudes(perf_range>0))/sum(amplitudes(:));
                A_pvs = sum(amplitudes(pvs_range>0));
                A_par = sum(amplitudes(par_range>0));
                A_perf = sum(amplitudes(perf_range>0));  

                D_par(XX,YY,SL) = sum(Dspace(par_range>0) .* amplitudes(par_range>0)') / A_par;
                D_pvs(XX,YY,SL)= sum(Dspace(pvs_range>0) .* amplitudes(pvs_range>0)') / A_pvs;
                D_perf(XX,YY,SL) = sum(Dspace(perf_range>0) .* amplitudes(perf_range>0)') / A_perf;

                if f_pvs == 0; D_pvs(XX,YY,SL) = NaN; end
                if f_perf == 0; D_perf(XX,YY,SL)= NaN; end

                d = A_par*E1pvs*E1perf*E2pvs*E2perf + A_pvs*E1par*E1perf*E2par*E2perf + A_perf*E1par*E1pvs*E2par*E2pvs;
                f_pvsCorr(XX,YY,SL) = A_pvs*E1par*E1perf*E2par*E2perf / d;
                f_blCorr(XX,YY,SL) = A_perf*E1par*E1pvs*E2par*E2pvs / d;   		    
            else
                D_perf(XX,YY,SL) = 0;
                D_par(XX,YY,SL) = 0;
                D_pvs(XX,YY,SL) = 0;
                f_blCorr(XX,YY,SL) = 0;
                f_pvsCorr(XX,YY,SL) = 0;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the parametric maps (D,D* and f) in the specified output folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nii.img = permute(D_par(end:-1:1,end:-1:1,:),[2 1 3]);
save_untouch_nii(nii,[output_folder '/' output_name '_D_par.nii'])

nii.img = permute(D_perf(end:-1:1,end:-1:1,:),[2 1 3]);
save_untouch_nii(nii,[output_folder '/' output_name '_D_perf.nii'])

nii.img = permute(D_pvs(end:-1:1,end:-1:1,:),[2 1 3]);
save_untouch_nii(nii,[output_folder '/' output_name '_D_pvs.nii'])

nii.img = permute(f_blCorr(end:-1:1,end:-1:1,:),[2 1 3]);
save_untouch_nii(nii,[output_folder '/' output_name '_f_bl.nii'])

nii.img = permute(f_pvsCorr(end:-1:1,end:-1:1,:),[2 1 3]);
save_untouch_nii(nii,[output_folder '/' output_name '_f_pvs.nii'])

save([output_folder '/' output_name '_D.mat'],'D_par')
save([output_folder '/' output_name '_D_star.mat'],'D_perf')
save([output_folder '/' output_name '_f.mat'],'f_blCorr')    

end
