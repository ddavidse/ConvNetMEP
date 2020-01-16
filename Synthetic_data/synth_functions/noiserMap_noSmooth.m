function out_array = noiserMap_noSmooth(in_array, amp)
% Dmytro Kolenov 
% 7th of June 2019
out_array = zeros(size(in_array));

for qwe = 1:size(in_array,1)

    profile = in_array(qwe,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  White Noise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % randn() generates random numbers that follow a Gaussian distribution.
    noised  = profile + amp.*randn(1, length(profile));

    % outProf = moveaver(noised);

    out_array(qwe,:) = noised;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  1/f
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fv = linspace(0, 1, 20);                                % Normalised Frequencies
a = 1./(1 + fv*2);                                      % Amplitudes Of ‘1/f’
b = firls(42, fv, a);                                   % Filter Numerator Coefficients
%figure(1)
%freqz(b, 1, 2^17)                                       % Filter Bode Plot

% Define the length of the vector for noise (numel of patch
N = numel(out_array);

ns = rand(1, N);

% Time domain noise 
invfn = filtfilt(b, 1, ns);                             % Create ‘1/f’ Noise
scaler  = 10;
invfn =  invfn/scaler;
% lets also add 1/f noise to the problem
out_array = out_array + reshape(invfn, size(out_array));

oneoverF = reshape(invfn, size(out_array));
% 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Photon Noise
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QE              = 0.60;
% ObjectMagnitude = 0;
% PupilDiameter   = 4.2;
% Exposuretime    = 1e-4;
% BandWidth       = 88;
% %Calculate number of photons in reaching detector
% nFot = 1000/(10^(ObjectMagnitude/2.5)); %number of photons from the object (fot/s/cm2/Angstrom)
% CollectorAreaCm2=pi*((100*PupilDiameter/2)^2); %CollectorArea in cm^2
% nFot = round(nFot*Exposuretime*CollectorAreaCm2*BandWidth); 
% nFot=nFot*QE;%total photons reaching the detector.
%  
% %BackGround = zeros(1,numel(outmap));
% BackGround = poissrnd(nFot*oneoverF/sum(oneoverF(:)))/100;
% outmap = outmap + BackGround;
%










% Weird stuff :D
% %% Ok let's consider differnet noise sources
% PhotonNoise=1; ReadNoise=1; DarkCurrent=0.01;

% %Introduction of Read Noise
% if ReadNoise==1 && flag==1
%     BackGround = readNoise(BackGround,SH,Exposuretime);
%     for i=1:length(PSF)
%         PSF{i}=BackGround(ML.coor(i,1):ML.coor(i,2), ML.coor(i,3):ML.coor(i,4));
%     end
% end

