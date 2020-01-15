% Heavily modified by Davy

clear all; close all; clc;
disp('Running...')

% Note: scanspeed is expected to be 3 digits

%% Inspect directory and make folders

% optional cd if data is in a different folder:
% cd('path of data folder')

a = dir;
b = struct2cell(a);
c = b';
d = c(:,1);
L = size(d,1);

maindir = pwd;

tag = 'smoothed_and_detrended';
if exist(tag,'dir') ~= 7
    mkdir(tag)
end
smoothedir = [pwd '\' tag];

tag = 'plots_raw';
if exist(tag,'dir') ~= 7
    mkdir(tag)
end
psavedir = [pwd '\' tag];

%% Convert TDMS to mat 

w = waitbar(0, 'Processing TDMS...');
w.Color = [1 1 0.4];
tic

count = 0;
tc = zeros(1,L);
tp = zeros(1,L);
ts = zeros(1,L);

for k = 1:L
    
    F = d{k};
        
    if (size(F,2) > 5) && (sum(F(end-4:end) == '.tdms') == 5)
            
        NB = F(1:end-5);
        MN = [NB, '.mat'];
            
        if exist(MN,'file') == 2
                
            disp('TDMS file found: no conversion, mat file already exists') 
        else
            disp('TDMS file found: converting...')
            tic
            convertTDMS(true, F);
            tc(k) = toc;
            count = count + 1;
            %disp('...done')
        end
    end
    
    waitbar(k/L,w)
    p = round(100*k/L);
    w.Name = [num2str(p) '%']; 
end

%% Process mat files

a = dir;
b = struct2cell(a);
c = b';
d = c(:,1);
L = size(d,1);

v = waitbar(0, 'Processing mat...');
v.Position = [584.2500 291.7500 270 56.2500];
v.Color = [1 1 0.4];

for k = 1:L
    
    F = d{k};
    
    if (size(F,2) > 5) && (sum(F(end-3:end) == '.mat') == 4)
        
        FNpart = F(1:5);
        OutName = strcat(FNpart, '_S_D.mat');
        cd(smoothedir)
        
        if exist(OutName,'file') ~= 2
        
            tic
        
            disp('mat file found: processing...')
            
            cd(maindir)
            data = load(F);
            data2 = data.ConvertedData.Data.MeasuredData;
            data_cell = struct2cell(data2);
            Signal = data_cell{10};
            Trigger_data = data_cell{14};

            Trigger_better = round(Trigger_data/max(Trigger_data));

            I = regexp(F,'Nsamplesperline');
            nsamples_pre = F(I+16:I+23);
            nsamples = str2double(nsamples_pre);

            I = regexp(F,'Xspeed');
            scanspeed_pre = F(I+7:I+9);        
            scanspeed = str2num(scanspeed_pre) / 1000; %#ok<ST2NM>

            if size(scanspeed,1) == 0
                scanspeed_pre = F(I+7:I+8);        
                scanspeed = str2double(scanspeed_pre) / 1000;
            end

            I = regexp(F,'NLines');
            nlines_pre = F(I+7:I+11);
            nlines = str2num(nlines_pre); %#ok<ST2NM>

            if size(nlines,1) == 0                    
                nlines_pre = F(I+7:I+10);
                nlines = str2double(nlines_pre);
            end

            I = regexp(F,'LineSep');
            delta_pre = F(I+8:I+12);
            delta   = str2double(delta_pre) * 1000;

            downsample = 1;

            outdata =  plotscanTrigMAT(Signal,Trigger_better,nsamples,nlines,'hide');

            downsampleMapLR = outdata(2:2:end,1:downsample:end);
            %downsampleMapRL = outdata(2:2:end,1:downsample:end);

            Xmax = size(downsampleMapLR,2);
            Xremove = round(Xmax/3);
            D1 = downsampleMapLR(:,Xremove:end);

            %figure('Position',[47 313 866 532],'Visible','Off')
            figure('Position',[1 41 1920 964],'Visible','Off')
            s1 = subplot(1,1,1);
            image(D1,'CDataMapping','scaled');colormap jet;
            s1.XLim = [0 size(D1,2)]; s1.YDir = 'normal';

            cd(psavedir)
            NB = F(1:end-5);
            savetag = [NB '.png'];        
            saveas(gcf, savetag)
            cd ..
            tp(k) = toc;

            close


            % smoothe and detrend

            tic

            V = size(D1,1);
            H = size(D1,2);

            C = zeros(V,H);

            for i = 1:H

                D = D1(:,i);

                for j = 6:V-5
                    %C(j,i) = (D(j) + D(j-1) + D(j+1) + D(j-2) + D(j+2) + D(j-3) + D(j+3))/7;
                
                    C(j,i) = (D(j) + D(j-1) + D(j+1) + D(j-2) + D(j+2) + D(j-3) + D(j+3) + D(j-4)...
                    + D(j+4) + D(j+5) + D(j-5))/11;
                
                    %C(j,i) = (D(j) + D(j-1) + D(j+1) + D(j-2) + D(j+2) + D(j-3) + D(j+3) + D(j-4)...
                    %+ D(j+4) + D(j+5) + D(j-5) + D(j+6) + D(j-6) + D(j+7) + D(j-7))/15;
                end

                C(1,i) = (D(1) + D(2) + D(3) + D(4) + D(5))/5;
                C(2,i) = (D(1) + D(2) + D(3) + D(4) + D(5) + D(6))/6;
                C(3,i) = (D(1) + D(2) + D(3) + D(4) + D(5) + D(6) + D(7))/7;
                C(4,i) = (D(1) + D(2) + D(3) + D(4) + D(5) + D(6) + D(7) + D(8))/8;
                C(5,i) = (D(1) + D(2) + D(3) + D(4) + D(5) + D(6) + D(7) + D(8) + D(9))/9;

                C(V-4,i) = (D(V-8) + D(V-7) + D(V-6) + D(V-5) + D(V-4) + D(V-3) + D(V-2) + D(V-1) + D(V))/9;
                C(V-3,i) = (D(V-7) + D(V-6) + D(V-5) + D(V-4) + D(V-3) + D(V-2) + D(V-1) + D(V))/8;
                C(V-2,i) = (D(V-6) + D(V-5) + D(V-4) + D(V-3) + D(V-2) + D(V-1) + D(V))/7;
                C(V-1,i) = (D(V-5) + D(V-4) + D(V-3) + D(V-2) + D(V-1) + D(V))/6;
                C(V,i) = (D(V-4) + D(V-3) + D(V-2) + D(V-1) + D(V))/5;
            end

            C_T = C';
            C2_T = detrend(C_T);
            C2 = C2_T';

            C3 = detrend(C2);

            cd(smoothedir)
            save(OutName, 'C3');
            cd ..

            ts(k) = toc;
            
        else
            disp('mat file found: SD file already exists')
        end
    end
    
    waitbar(k/L,v)
    p = round(100*k/L);
    v.Name = [num2str(p) '%']; 

end
    
disp(' ')
disp(['Time of conversion: ', num2str(sum(tc)), ' s'])
disp(['Amount of files converted: ', num2str(count)])
disp(' ')
disp(['Time of processing and plotting: ', num2str(sum(tp)), ' s'])
disp(['Time of smoothing and detrending: ', num2str(sum(ts)), ' s'])


%% Save Plots 

cd(smoothedir)
tag = 'plots';
if exist(tag,'dir') ~= 7
    mkdir(tag)
end
SDPdir = [pwd '\' tag];

a = dir;
b = struct2cell(a);
c = b';
d = c(:,1);

l = size(d,1);

tic;

h = waitbar(0, 'Saving plots...');
h.Position = [584.2500 207 270 56.2500];
h.Color = [0.6 1 0.6];
plotvar = 0;

for i = 1:l
    
    if size(d{i},2) == 13
        
        plotvar = plotvar + 1;
        
        C = load(d{i}); %struct
        FieldNames = fieldnames(C); %cell with field name
        N = FieldNames{1}; %just the field name
        D1 = getfield(C,N); %just the data array

        %figure('Position',[47 313 866 532],'Visible','Off')
        figure('Position',[1 41 1920 964],'Visible','Off')
        s1 = subplot(1,1,1);
        image(D1,'CDataMapping','scaled');colormap jet;
        s1.XLim = [0 size(D1,2)]; s1.YDir = 'normal';
        
        cd(SDPdir)
        savetag = ['plot_' num2str(plotvar) '.png'];        
        saveas(gcf, savetag)
        cd(smoothedir)
        
        close
        
        
    end
    
    waitbar(i/l,h)
    q = round(100*i/l);
    h.Name = [num2str(q) '%']; 
    
end

delete(w)
delete(v)
delete(h)
t = toc;
disp(' ')
disp(['Time of saving SD plots: ' num2str(t) ' s'])
cd(maindir)

t_tot = sum(tc) + sum(tp) + sum(ts) + t;
disp(['Total time elapsed: ', num2str(t_tot), ' s'])
winopen(SDPdir)










