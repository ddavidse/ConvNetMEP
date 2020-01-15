function sweep =  plotscanTrigMAT(signal, trigger, nsamples, nlines, prop, k)

if nargin < 4
    k = 1;
    prop = 'hide';
end

% Dmytro Kolenov 29/11/2017 d.kolenov@tudelft.nl
% function sweep =  plotscan(filename,nsamples,nlines,partnumber)

if strcmp(prop,'show')
    figure(), plot(signal)
    hold on, plot(trigger)
end

signalmatrix = reshape(signal,[nsamples nlines]);

if strcmp(prop,'show')
    figure(), surf(signalmatrix'), shading interp, view(0,90)
    figure(1), title('\bf Raw data')
end

% NB -> Not always finds the rising edge of the signal at this point
thresh = 0.3*max(trigger(:));
[peakValues, indexes] = findpeaks(trigger, 'MinPeakHeight', thresh, 'MinPeakDistance', 0.95*nsamples);

if strcmp(prop, 'show')
    figure(), plot(trigger(1:k*nsamples));
    hold on, plot(indexes(1:k), peakValues(1:k), '*', 'MarkerSize', 8);
end

stockInd = cell(1, nlines);
stockVal = cell(1, nlines);
finalMap = zeros(nlines, nsamples);

marginVal = 600;
grid = 1:(nsamples + marginVal);

iii = 1;
% Probably +1

for zzz = 1:1:(length(indexes) - 1)
    
    pair = [indexes(zzz),indexes(zzz+1)];
    % grid = 1:(indexes(zzz+1)-indexes(zzz));

    if strcmp(prop,'show')
    disp(pair)
    end

    indexrow = pair(1):pair(2);
    correspondVal = signal(indexrow);
    stockInd{iii} = indexrow;
    stockVal{iii} = correspondVal;

    % Whats wrong with matlabs interpolation function ?
    % figure(),plot(correspondVal);
    % afterinterp   = interp1(indexrow, correspondVal', grid,'linear');
    % afterinterp   = afterinterp(~isnan(afterinterp));
    % figure(),plot(afterinterp);
    % finalMap(iii,:) = afterinterp;

    if length(correspondVal) > nsamples
        correspondVal = correspondVal(1:nsamples);
    elseif length(correspondVal) < nsamples 
        discrep = nsamples - length(correspondVal);
        correspondVal = padarray(correspondVal,[discrep 0],0,'pre');
    else
        string = 'Array is suitable';
    end
        
    finalMap(iii,:) = correspondVal;
    iii = iii + 1;
end


% newmap = 
% a = 123;

% You reshape the array to fit to the amount of samples made along the X
% direction 
% The only parameter that you need to care about when transporting data
% from experimental setup to PC is nsamples

% One can also do it with nlines and transposition of matrix however the
% sweeps that belong to one session might stack in the single file and you 
% won't have excatly the same amount of lines that you've asked in 2D scan
% newfile = newfile((partnumber+1):(partnumber+nlines*nsamples));
% matrix = reshape (newfile,nlines,nsamples);
% matrix = reshape (newfile,length(newfile)/nsamples,nsamples);
 
% matrix = matrix(1:nlines,1:(nsamples/factor));
% figure()
% surf(matrix)
% xlabel('\bf X samples','FontSize',14)
% ylabel('\bf Y lines','FontSize',14)
% zlabel('\bf Diff. Scan signal [v]','FontSize',16)
% colorbar; shading interp;

sweep = finalMap;

end


