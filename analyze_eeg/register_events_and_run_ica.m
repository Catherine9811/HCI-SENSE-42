% EEGLAB history file generated on the 16-Feb-2025
% ------------------------------------------------
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadset('filename','P001.set','filepath','C:\\Users\\MartinBai白心宇\\PycharmProjects\\HCI\\data\\EEG\\');
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:32] ,'computepower',1,'linefreqs',50,'newversion',0,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',0,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','P001 Remove Power Line','savenew','C:\\Users\\MartinBai白心宇\\PycharmProjects\\HCI\\data\\EEG\\P001_RM_PowerLine.set','gui','off'); 
figure; pop_spectopo(EEG, 1, [0      7894998.0469], 'EEG' , 'percent', 20, 'freq', [6 22 50], 'freqrange',[1 100],'electrodes','off');
figure; pop_spectopo(EEG, 1, [0      7894998.0469], 'EEG' , 'freq', [6 22 50], 'freqrange',[1 100],'electrodes','off');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'retrieve',1,'study',0); 
EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:32] ,'computepower',1,'linefreqs',[50 100] ,'newversion',0,'normSpectrum',0,'p',0.05,'pad',2,'plotfigures',0,'scanforlines',0,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','P001 Remove Power Line Noise','savenew','C:\\Users\\MartinBai白心宇\\PycharmProjects\\HCI\\data\\EEG\\P001_RM_PowerLine.set','gui','off'); 
figure; pop_spectopo(EEG, 1, [0      7894998.0469], 'EEG' , 'freq', [50], 'freqrange',[1 120],'electrodes','off');
pop_selectcomps(EEG, [1:31] );
EEG = pop_saveset( EEG, 'savemode','resave');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = pop_subcomp( EEG, [], 0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'setname','P001 Remove Power Line Noise Pruned with ICA','savenew','C:\\Users\\MartinBai白心宇\\PycharmProjects\\HCI\\data\\EEG\\P001_RM_PowerLine_ICA.set','gui','off'); 
figure; pop_spectopo(EEG, 1, [0      7894998.0469], 'EEG' , 'freq', [6 10 22], 'freqrange',[1 100],'electrodes','off');
EEG = pop_saveset( EEG, 'savemode','resave');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;
