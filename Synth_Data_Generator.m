% Dmytro Kolenov, Davy Davidse
% d.kolenov@tudelft.nl
% d.j.davidse@student.tudelft.nl

clear all; close all; clc
tic

addpath('D:\ddavidse\Desktop\Synthesize\functions')
savedir = 'D:\ddavidse\Desktop\comparison test';

PixelCount = 150; % number of pixels
N = 131; % number of images per class

widthBeforeCut   = PixelCount + 50; % X dimension
heightBeforeCut  = PixelCount + 50; % Y dimension

% Recalculate to the size of the rescaled pattern
inL = [70, 87, 96];%, 103, 112];
setOfNlines = floor(inL*PixelCount/400); % 6 6 4 

inW = [43, 76, 112];%, 134, 161];
setOfHalfW  = inW / 400;
setOfAmp     = [0.4, 0.8, 1.2];%, 1.6, 2];

% Smaller particles you can shift more
RangeShiftTopVal = [55, 50, 45];%, 40, 35];

% "Scaler" is the parameter that defines rounding; should not exceed 500 and should be
% defined carefully 
% In principal more lines means lesser stepper for scaler
% Parameter defines how far we step in gaussin between points
% setOfScalers = [15,12,10,8];
setOfScalers   = floor(fliplr(setOfNlines)./20);

y_flip = 1;
x_flip = -1;

cd(savedir)
mkdir('Data')
mkdir('Images')
datadir = [pwd '\Data'];
imagedir = [pwd '\Images'];

for i = 1:length(setOfAmp)
    
    w = waitbar(0, ['Synthesizing: class ' num2str(i)]);
    w.Color = [1 1 0.4];
    
    cd(datadir)
    addpart = ['Class ' num2str(i)];
    mkdir(addpart)
    datasavedir = [pwd '\' addpart];
    
    cd(imagedir)
    mkdir(addpart)
    imagesavedir = [pwd '\' addpart];
    
    
    % Every second image is inverse order (min and then maxima)
    for j = 1:N
        
        PeakAmp = setOfAmp(i);
        HalfDur = setOfHalfW(i);
        nlines  = setOfNlines(i);   
        scaler  = setOfScalers(i);
        
        % alternative map can be created with the smooth peak and
        % p_wav function and its counterpart
        % 671 page of Fundamentals of Statistical Signal Processing, Volume III
        % x=0.01:0.01:2;
        % PulseWidth Function think of making the signal wider
        % PeakAmp = 2;
        % HalfDur = 0.2;
        
        x  = linspace(0, 2, widthBeforeCut);
        % x  = linspace(0, HalfDur*10, width);

        xSamples  = 1:widthBeforeCut;
        ySamples  = 1:heightBeforeCut;
        li = 72/72;  

        ScanMapSig = zeros(length(x), nlines);

        [pointsOut] = returnFromNormal(nlines, scaler);

        % At this point it still goes nice in terms of 4.2 being approximately
        % this value
        amprange = pointsOut*PeakAmp; % where const is a peak value in the middle of scan
        % only half duration right
        durrange = pointsOut*HalfDur;
        
        for k = 1:size(ScanMapSig,2)
        % period is pretty much period = (d_qrswav + d_swav);

            ampleft    = amprange(k);
            durleft    = durrange(k);
            durbetween = 0.1;
            ampright   = ampleft;
            durright   = durleft;

            % If amplitude is not equal than there is kind of DC offset
            shift = floor(length(x)/2);
    
            % Here is where it goes wrong for the amplitude of 4.2!
            [resultingSig,period]= generateDiff(x,li,ampleft,durleft,durbetween,ampright,durright,shift);
      
            % figure(), plot(x, circshift(resultingSig, floor(length(resultingSig)/2)))%
            ScanMapSig(:,k) = resultingSig;
        end
        
        ScanMapSig = ScanMapSig';
    
        % To compensate for peak value not 4.2
        peakVal = max(ScanMapSig(:));
        magCor = PeakAmp/peakVal; 
        ScanMapSig = magCor.*ScanMapSig;
    
        RemainderToPad = heightBeforeCut - nlines;
        halfPad = floor(RemainderToPad/2);
        isenough = halfPad*2 + nlines;

        if heightBeforeCut > isenough
        
            additionalpiece = zeros((heightBeforeCut-isenough), widthBeforeCut);
            paddedMap = padarray(ScanMapSig, [halfPad,0], 0, 'both');
            paddedMap = [paddedMap; additionalpiece];
    
        else
            paddedMap = padarray(ScanMapSig, [halfPad,0], 0, 'both');
        end

        % Introduce noise to the problem 
        paddedMapN = noiserMap_noSmooth(paddedMap, 1e-2);

        string = [num2str(i),'_iteration._',num2str(j/10),'_',num2str(PeakAmp),'_Amplitude_And_',num2str(num2str(HalfDur)),'_Width'];
        % title(['Peak amplitude of signal ',num2str(PeakAmp),' And width ', num2str(HalfDur)])
         
        % % Very important to save 
        % I = paddedMap - min(paddedMap(:));
        % I = paddedMap ./ max(paddedMap(:));
        % [J,~] = gray2ind(I);
        % imwrite(J, jet, string)
        
        if rem(j,2)
            
            paddedMapN = fliplr(paddedMapN);                                % flip particle along x-axis
        end
    
        yy = randi(RangeShiftTopVal(i));
        xx = randi(RangeShiftTopVal(i));
               
        shiftMap_0 = vertCirc(paddedMapN, y_flip*yy); 
        shiftMap = horCirc(shiftMap_0, x_flip*xx);                          % randomly move particle
        shiftMap = shiftMap(26:(end-25), 26:(end-25));                      % remove extra space
        
        % disp(['x_flip*xx = ' num2str(x_flip*xx)])
        % disp(['x_flip = ', num2str(x_flip)])
   
        % Here is the part crucial for the DL 
        % notice the range of generated values!
        % 1/f noise is giving the DC offset from -- noiserMap_noSmooth
          
        MaxConstraint = 10;
        shiftMap = shiftMap./MaxConstraint;
        min(shiftMap(:));
        max(shiftMap(:));
        
        %     if ~rem(eee,2)
        %     A = imagesc(shiftMap);colormap gray
        %     I = A.CData;
        %     else
        %     A = imagesc(shiftMap);colormap gray
        %     I = fliplr(A.CData);
        %     end
        %     close all;
        
        I = mat2gray(shiftMap);
        MATI = shiftMap;
    
        cd(imagesavedir)
        imagestring = [string, '.jpg'];
        imwrite(I, imagestring)
        newmap = gray;
        
        cd(datasavedir)
        save([num2str(i),'_iteration._',num2str(j/10),'_',num2str(PeakAmp),'_Amplitude_And_',num2str(num2str(HalfDur)),'_Width.mat'],'MATI')
    
        x_flip = -1*x_flip;
        y_flip = -1*y_flip;
        
        waitbar(j/N, w)
        p = round(100*j/N);
        w.Name = [num2str(p) '%'];
    end
    
    delete(w)
    deb = 'debugging';
    
%     f = figure('Position',[431 143 1091 772]);
%     f.Name = ['Class ' num2str(i)];
%
%     subplot(2,2,1)
%     surf(ScanMapSig);shading interp; colormap jet; view(0,90);
%     title('ScanMapSig')
%     subplot(2,2,2)
%     surf(paddedMap);shading interp; colormap jet; view(0,90);
%     title('paddedMap')
%     subplot(2,2,3)
%     surf(paddedMapN);shading interp; colormap jet; view(0,90);
%     title('paddedMapN')
%     subplot(2,2,4)
%     surf(shiftMap);shading interp; colormap jet; view(0,90);
%     title('shiftMap')
end

cd(savedir)
disp('Data synthesizing complete')
t = toc;
disp(['Elapsed time: ' num2str(t) ' s'])



