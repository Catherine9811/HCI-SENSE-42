#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.3),
    on February 18, 2025, at 11:32
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from global_code
import re
import cv2
import threading, time
from psychopy.visual.movies import MovieFrame
from psychopy import logging
logging.console.setLevel(logging.CRITICAL)


from faker import Faker

fake = Faker(["en_US"], use_weighting=True)

personalized_settings = {
    "typing_speed": 3.13, # characters per seconds
}

import serial

class SerialConnector:
    CALIB_BEGIN = 1
    CALIB_END = 2
    
    EXP_BEGIN = 3
    EXP_END = 4
    
    WINDOW_CLOSE_BEGIN = 5
    WINDOW_CLOSE_END = 6
    
    # Mail related events
    MAIL_HOMESCREEN_BEGIN = 7
    MAIL_HOMESCREEN_END = 8
    MAIL_NOTIFICATION_BEGIN = 9
    MAIL_NOTIFICATION_END = 10
    MAIL_CONTENT_BEGIN = 11
    MAIL_CONTENT_END = 12
    
    # File manager (dragging and opening)
    FILE_MANAGER_HOMESCREEN_BEGIN = 13
    FILE_MANAGER_HOMESCREEN_END = 14
    FILE_MANAGER_DRAGGING_BEGIN = 15
    FILE_MANAGER_DRAGGING_END = 16
    FILE_MANAGER_OPENING_BEGIN = 17
    FILE_MANAGER_OPENING_END = 18
    
    # Trash bin
    TRASH_BIN_HOMESCREEN_BEGIN = 19
    TRASH_BIN_HOMESCREEN_END = 20
    TRASH_BIN_SELECT_BEGIN = 21
    TRASH_BIN_SELECT_END = 22
    TRASH_BIN_CONFIRM_BEGIN = 23
    TRASH_BIN_CONFIRM_END = 24
    
    # Notes
    NOTES_HOMESCREEN_BEGIN = 25
    NOTES_HOMESCREEN_END = 26
    NOTES_REPEAT_BEGIN = 27
    NOTES_REPEAT_END = 28
    
    # Browser
    BROWSER_HOMESCREEN_BEGIN = 29
    BROWSER_HOMESCREEN_END = 30
    BROWSER_NAVIGATION_BEGIN = 31
    BROWSER_NAVIGATION_END = 32
    BROWSER_CONTENT_BEGIN = 33
    BROWSER_CONTENT_END = 34
    
    # Questions
    QUESTION_BASE = 100
    QUESTION_LEAP = 10
    ## Answered questions will be encoded as 
    ## QUESTION_BASE + QUESTION_LEAP * QUESTION_INDEX + QUESTION_RATING
    
    EEG_STOP_RECORDING = 255
    EEG_START_RECORDING = 254

    def __init__(self, com: str, bit_rate: int):
        try:
            self.port = serial.Serial(com, bit_rate, write_timeout=0.1, timeout=0.1)
        except Exception as e:
            self.port = None
            logging.critical("EEG recording is not enabled!")
            logging.critical(e)
    
    def write(self, number):
        if self.port is not None:
            try:
                self.port.write(int(number).to_bytes(1, 'big'))
            except Exception as e:
                logging.critical(f"Failed to send EEG signal {number}!")
                logging.critical(e)
        else:
            logging.critical(f"Virtually sending signal {number} to EEG")
    
    def open(self):
        if self.port is not None:
            try:
                self.port.open()
            except Exception as e:
                logging.critical(f"Failed to open EEG port!")
                logging.critical(e)
                
    def close(self):
        if self.port is not None:
            try:
                self.port.close()
            except Exception as e:
                logging.critical(f"Failed to close EEG port!")
                logging.critical(e)
serial_connector = SerialConnector('COM3', 115200)
serial_connector.open()
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.3'
expName = 'explorer'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 960]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\MartinBai白心宇\\Documents\\Personal\\HCI\\Computer\\explorer_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color='white', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('calibration_start_key') is None:
        # initialise calibration_start_key
        calibration_start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='calibration_start_key',
        )
    if deviceManager.getDevice('calibration_typing_start_key') is None:
        # initialise calibration_typing_start_key
        calibration_typing_start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='calibration_typing_start_key',
        )
    if deviceManager.getDevice('calibration_typing_key') is None:
        # initialise calibration_typing_key
        calibration_typing_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='calibration_typing_key',
        )
    if deviceManager.getDevice('calibration_end_key') is None:
        # initialise calibration_end_key
        calibration_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='calibration_end_key',
        )
    if deviceManager.getDevice('experiment_start_key') is None:
        # initialise experiment_start_key
        experiment_start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='experiment_start_key',
        )
    if deviceManager.getDevice('mail_content_user_key_release') is None:
        # initialise mail_content_user_key_release
        mail_content_user_key_release = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='mail_content_user_key_release',
        )
    if deviceManager.getDevice('notes_repeat_keyboard') is None:
        # initialise notes_repeat_keyboard
        notes_repeat_keyboard = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='notes_repeat_keyboard',
        )
    if deviceManager.getDevice('browser_navigation_user_key_release') is None:
        # initialise browser_navigation_user_key_release
        browser_navigation_user_key_release = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='browser_navigation_user_key_release',
        )
    if deviceManager.getDevice('experiment_end_key') is None:
        # initialise experiment_end_key
        experiment_end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='experiment_end_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # enter 'rush' mode (raise CPU priority)
    core.rush(enable=True)
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "definition" ---
    # Run 'Begin Experiment' code from global_code
    FONT_SIZE = 0.1
    
    class VideoRecorder:
        def __init__(self, camera_index=0):
            self.camera_index = camera_index
            self.cap = None
            self.is_recording = False
            self.out = None
            self.thread = None
            self.latest_frame = None
            self.frame_size = (1280, 720)
            self.frame_rate = 30
            self.camRecFolder = thisExp.dataFileName + '_cam_recorded'
            if not os.path.isdir(self.camRecFolder):
                os.mkdir(self.camRecFolder)
    
        def open(self):
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
    
        def record(self):
            if self.cap is None:
                raise Exception("Camera is not opened")
    
            if self.is_recording:
                raise Exception("Already recording")
    
            self.is_recording = True
            # Save cam recording
            camFilename = os.path.join(
                self.camRecFolder, 
                'recording_cam_%s.mp4' % data.utils.getDateStr()
            )
            thisExp.addData('camera.filename', camFilename)
            self.thread = threading.Thread(target=self._record_loop, args=(camFilename,), daemon=True)
            self.thread.start()
    
        def _record_loop(self, filename):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(filename, fourcc, self.frame_rate, self.frame_size)
    
            while self.is_recording and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(1 / self.frame_rate * 0.1)
                    continue
                self.out.write(frame)
                # if we have a new frame, update the frame information
                videoFrameArray = np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).flatten(), dtype=np.uint8)
                # provide the last frame
                self.latest_frame = MovieFrame(
                    frameIndex=0,
                    absTime=0.0,
                    # displayTime=self._recentMetadata['frame_size'],
                    size=self.frame_size,
                    colorFormat='rgb24',  # converted in thread
                    colorData=videoFrameArray,
                    audioChannels=0,
                    audioSamples=None,
                    metadata=None,
                    movieLib="opencv",
                    userData=None)
                time.sleep(1 / self.frame_rate * 0.5)
    
            self.out.release()
    
        def getVideoFrame(self):
            return self.latest_frame
    
        @property
        def frameSize(self):
            return self.frame_size
    
        def save(self):
            if self.is_recording:
                self.is_recording = False
                self.thread.join()
    
            if self.out:
                self.out.release()
    
        def list(self):
            index = 0
            available_cameras = []
            while True:
                cap = cv2.VideoCapture(index)
                if not cap.read()[0]:
                    break
                available_cameras.append(index)
                cap.release()
                index += 1
            return available_cameras
    
        def close(self):
            if self.is_recording:
                self.is_recording = False
                self.thread.join()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    camera_connector = VideoRecorder()
    camera_connector.open()
    
    
    class Stylizer:
        STYLES = ["windows", "mac"]
        predefined_style = None
        current_style = None
        
        def __init__(self, style=None):
            self.predefined_style = style
            self.reset(style)
            
        def log(self):
            thisExp.addData("operating_system_style", self.current_style)
                
        def reset(self, style=None):
            if self.predefined_style is not None:
                self.current_style = self.predefined_style
            elif style is None or style not in self.STYLES:
                self.current_style = np.random.choice(self.STYLES)
            else:
                self.current_style = style
    
    class Component:
        def __init__(self):
            self.COMPONENTS = []
            
        def hide(self):
            for component in self.COMPONENTS:
                component.setAutoDraw(False)
        
        def show(self):
            for component in self.COMPONENTS:
                component.setAutoDraw(True)
                component.draw()
    
    class TaskBar(Component):
        ICON_SIZE = 0.1
        ICON_SIZE_PRESSED = 0.09
        ICON_GAP = 0.005
        BAR_SIZE = 0.1
        FONT_SIZE = 0.1
        ASPECT_RATIO = win.size[0] / win.size[1]
        ICONS = {
            "File Manager": "resources/{}/explorer.png",
            "Browser": "resources/{}/network.png",
            "Mail": "resources/{}/mail.png",
            "Document": "resources/{}/notes.png",
            "Trash": "resources/{}/trash_full.png",
        }
        HOME = 'resources/{}/homescreen.png'
        TASKBAR = 'resources/{}/taskbar.png'
        HIGHLIGHT = 'resources/{}/highlight.png'
        
        icon_objects = []
        highlight_object = None
        target_name = None
        mouse = event.Mouse(win=win)
        
        def __init__(self, style, target=None):
            super().__init__()
            self.style = style
            self.allocate_target(index=target)
            self.draw_interface()
        
        def update(self):
            self.updateHighlight()
            return self.updateStatus()
            
        def allocate_target(self, index=None):
            if index is None or index >= len(self.ICONS):
                self.target_name = np.random.choice(list(self.ICONS.keys()))
            else:
                self.target_name = list(self.ICONS.keys())[index]
    
        def get_taskbar_icon_at(self, i):
            relative_location = i - len(self.ICONS) / 2
            relative_sign = np.sign(relative_location)
            absolute_location = np.abs(relative_location)
            return ((self.ICON_GAP + self.ICON_SIZE / self.ASPECT_RATIO) * relative_location, -1.0 + self.BAR_SIZE / 2)
        
        def reset(self):
            style = self.style.current_style
            for component in self.COMPONENTS:
                if isinstance(component, visual.ImageStim):
                    component.setImage(component.name.format(style))
                    
        def set_trash_empty(self):
            for component in self.COMPONENTS:
                if component.name == self.ICONS["Trash"]:
                    component.setImage(component.image.replace("full", "empty"))
        
        def draw_interface(self):
            style = self.style.current_style
            bg_image = visual.ImageStim(
                win=win,
                name=self.HOME, 
                units='norm', 
                image=self.HOME.format(style), anchor='center',
                pos=(0, -1 / 2 * self.FONT_SIZE), size=(2.0, 2.0 - self.FONT_SIZE),
                interpolate=True)
            bg_image.setAutoDraw(True)
            self.COMPONENTS.append(bg_image)
            task_bar = visual.ImageStim(
                win=win,
                name=self.TASKBAR,
                units="norm",
                image=self.TASKBAR.format(style),
                opacity=1.0,
                size=(2.0, self.BAR_SIZE + 0.005),
                pos=(0, -1.0),
                anchor="bottom-center"
            )
            task_bar.setAutoDraw(True)
            self.COMPONENTS.append(task_bar)
            self.highlight_object = visual.ImageStim(
                win=win,
                name=self.HIGHLIGHT, 
                units='norm', 
                image=self.HIGHLIGHT.format(style), anchor='center',
                pos=(0, 2.0), size=(self.ICON_SIZE / self.ASPECT_RATIO, self.ICON_SIZE),
                interpolate=True, opacity=0.1)
            self.highlight_object.setAutoDraw(True)
            self.COMPONENTS.append(self.highlight_object)
            for index, icon_resource in enumerate(self.ICONS.values()):
                icon = visual.ImageStim(
                    win=win,
                    name=icon_resource, 
                    units='norm', 
                    image=icon_resource.format(style), anchor='center',
                    pos=self.get_taskbar_icon_at(index + 1), size=(self.ICON_SIZE / self.ASPECT_RATIO, self.ICON_SIZE),
                    interpolate=True)
                self.icon_objects.append(icon)
                icon.setAutoDraw(True)
                self.COMPONENTS.append(icon)
            pass
    
        def updateHighlight(self):
            if self.highlight_object is None:
                return
            for element in self.icon_objects:
                if element.contains(self.mouse.getPos()):
                    self.highlight_object.setPos(element.pos)
                    return
            self.highlight_object.setPos((0, 1.0))
            return
        
        def updateStatus(self):
            for element, element_name in zip(self.icon_objects, self.ICONS.keys()):
                if self.mouse.isPressedIn(element):
                    element.setSize((self.ICON_SIZE_PRESSED / self.ASPECT_RATIO, self.ICON_SIZE_PRESSED))
                    if element_name == self.target_name:
                        return False
                else:
                    element.setSize((self.ICON_SIZE / self.ASPECT_RATIO, self.ICON_SIZE))
            return True
            
    class BaseWindow(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        ASPECT_RATIO = win.size[0] / win.size[1]
        NAME = "Window"
        TOOLS = {
            "windows": {
                "Close": "resources/windows/close.png",
                "Maximize": "resources/windows/max.png",
                "Minimize": "resources/windows/min.png",
            },
            "mac": {
                "Close": "resources/mac/close.png",
                "Minimize": "resources/mac/min.png",
                "Maximize": "resources/mac/max.png",
            }
        }
        TOOLS_HIGHLIGHT = {
            "windows": {
                "Close": "resources/windows/close_highlight.png",
                "Maximize": "resources/windows/max_highlight.png",
                "Minimize": "resources/windows/min_highlight.png",
            },
            "mac": {
                "Close": "resources/mac/close_highlight.png",
                "Minimize": "resources/mac/min_highlight.png",
                "Maximize": "resources/mac/max_highlight.png",
            }
        }
        
        mouse = event.Mouse(win=win)
        objects = {}
        
        def __init__(self, style):
            super().__init__()
            self.style = style
            self.allocate_target()
            self.draw_interface()
            
        def update(self):
            return self.updateHighlight()
        
        def log(self):
            thisExp.addData("window_close_target_name", self.target_name)
            
        def update_title(self, name):
            self.NAME = name
            self.title.text = name
            
        def allocate_target(self, index=None):
            if index is None or index >= len(self.TOOLS[self.style.current_style]):
                self.target_name = np.random.choice(["Close", "Minimize"])
            else:
                self.target_name = list(self.TOOLS[self.style.current_style].keys())[index]
        
        def reset(self):
            style = self.style.current_style
            self.title.alignment = self.calculate_title_layout(style)
            for element_name, element_tuple in self.objects.items():
                for element, definition in zip(element_tuple, [self.TOOLS, self.TOOLS_HIGHLIGHT]):
                    element.setImage(definition[style][element_name])
                    index = list(definition[style].keys()).index(element_name)
                    param_dict = self.calculate_tool_layout(index, style)
                    for key, value in param_dict.items():
                        setattr(element, key, value)
        
        def draw_interface(self):
            style = self.style.current_style
            bg_window = visual.Rect(
                win=win,
                name="base_window", 
                units='norm',
                fillColor=[0.8824, 0.8824, 0.8824], opacity=0.96, anchor='top-center',
                pos=(0, 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE), size=(2.0, 2.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.BAR_SIZE),
                interpolate=True)
            bg_window.setAutoDraw(True)
            self.COMPONENTS.append(bg_window)
            bg_toolbar = visual.Rect(
                win=win,
                name="base_toolbar", 
                units='norm',
                fillColor="lightgray", opacity=1.0, anchor='top-center',
                pos=(0, 1.0 - self.FONT_SIZE), size=(2.0, self.TOOLBAR_SIZE),
                interpolate=True)
            bg_toolbar.setAutoDraw(True)
            self.COMPONENTS.append(bg_toolbar)
            self.title = visual.TextBox2(
                 win, text=self.NAME, font='Open Sans',
                 pos=(-1.0 + self.FONT_SIZE / 4, 1.0 - self.FONT_SIZE), units='norm', letterHeight=self.FONT_SIZE / 2,
                 size=(2.0 - self.FONT_SIZE / 2, self.TOOLBAR_SIZE), borderWidth=2.0,
                 color='black', colorSpace='rgb',
                 opacity=None, bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.0, alignment=self.calculate_title_layout(style),
                 anchor='top-left', overflow='visible',
                 fillColor=None, borderColor=None,
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='base_window_title', 
                 autoLog=False
            )
            self.title.setAutoDraw(True)
            self.COMPONENTS.append(self.title)
            for index, (tool_name, tool_resource, tool_highlight_resource) in enumerate(zip(self.TOOLS[style].keys(), self.TOOLS[style].values(), self.TOOLS_HIGHLIGHT[style].values())):
                icon = visual.ImageStim(
                    win=win,
                    name=tool_name, 
                    units='norm', 
                    image=tool_resource, size=(self.BAR_SIZE / self.ASPECT_RATIO, self.BAR_SIZE),
                    interpolate=True,
                    **self.calculate_tool_layout(index, style))
                icon.setAutoDraw(True)
                icon_highlight = visual.ImageStim(
                    win=win,
                    name=f"{tool_name}_highlight", 
                    units='norm', 
                    image=tool_highlight_resource, size=(self.BAR_SIZE / self.ASPECT_RATIO, self.BAR_SIZE),
                    interpolate=True, opacity=0.0,
                    **self.calculate_tool_layout(index, style))
                icon_highlight.setAutoDraw(True)
                self.objects[tool_name] = (icon, icon_highlight)
                self.COMPONENTS.append(icon)
                self.COMPONENTS.append(icon_highlight)
        pass
        
        def calculate_tool_layout(self, index, style):
            params = {}
            if style == "windows":
                params["anchor"] = 'top-right'
                params["pos"] = (1.0 - index * self.BAR_SIZE / self.ASPECT_RATIO, 1.0 - self.FONT_SIZE)
            else:
                params["anchor"] = 'top-left'
                params["pos"] = (-1.0 + index * self.BAR_SIZE / self.ASPECT_RATIO, 1.0 - self.FONT_SIZE)
            return params
            
        def calculate_title_layout(self, style):
            if style == "windows":
                return 'center-left'
            return 'center'
        
        def updateHighlight(self):
            for element_name, element_tuple in self.objects.items():
                element = element_tuple[1]
                if element.contains(self.mouse.getPos()):
                    element.opacity = 1.0
                else:
                    element.opacity = 0.0
                if self.mouse.isPressedIn(element) and element_name == self.target_name:
                    return False
            return True
    
    
    class FormOverlay(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        SEARCHBAR_SIZE = 0.1
        SLIDER_SIZE = 0.1
        SLIDER_WIDTH = 1.4
        LABEL_WIDTH = 2.0
        BUTTON_WIDTH = 0.2
        ASPECT_RATIO = win.size[0] / win.size[1]
        SLIDERS = []
        HINTS = []
        
        current_index = 0
        order = []
        mouse = event.Mouse(win=win)
        
        def __init__(self, random=False):
            super().__init__()
            self.random = random
            self.draw_interface()
            self.randomize_order()
            
        def randomize_order(self):
            self.order = list(range(len(self.SLIDERS)))
            if self.random:
                np.random.shuffle(self.order)
            
        def update(self):
            display_index = self.order[self.current_index]
            if self.mouse.isPressedIn(self.next_button) and self.SLIDERS[display_index].getRating() is not None:
                serial_connector.write(SerialConnector.QUESTION_BASE + SerialConnector.QUESTION_LEAP * display_index + self.SLIDERS[display_index].getRating())
                self.current_index += 1
                if self.current_index >= len(self.SLIDERS):
                    self.current_index = 0
                    return False  # End of the form
            self.update_interface()  # Show next hint/slider
            return True
            
        def log(self):
            for slider_component in self.SLIDERS:
                thisExp.addData(f"{slider_component.name}.rating", slider_component.getRating())
                thisExp.addData(f"{slider_component.name}.rt", slider_component.getRT())
            
        def reset(self):
            self.current_index = 0
            for slider_component in self.SLIDERS:
                slider_component.reset()
            self.randomize_order()
            self.update_interface()
        
        def update_interface(self):
            display_index = self.order[self.current_index]
            for index, (hint, slider) in enumerate(zip(self.HINTS, self.SLIDERS)):
                if index == display_index:
                    hint.setAutoDraw(True)
                    hint.draw()
                    slider.setAutoDraw(True)
                    slider.draw()
                else:
                    hint.setAutoDraw(False)
                    slider.setAutoDraw(False)
            if self.current_index == len(self.SLIDERS) - 1:
                self.next_button.text = "Finish"
            else:
                self.next_button.text = "Next"
        
        def draw_interface(self):
            self.next_button = visual.TextBox2(win, text='Next', pos=(0, -0.3), font='Open Sans',
                     units='norm', color='black', letterHeight=self.FONT_SIZE / 2,
                     size=(self.BUTTON_WIDTH, self.SLIDER_SIZE), alignment='center',
                     anchor='top-center', borderWidth=1.0, borderColor="black")
            self.next_button.setAutoDraw(True)
            self.COMPONENTS.append(self.next_button)
            texts = [
                "Sleepiness: How sleepy are you?",
                "Mental Demand: How mentally demanding was the task?",
                #"Physical Demand: How physically demanding was the task?",
                "Temporal Demand: How hurried or rushed was the pace of the task?",
                "Performance: How successful were you in accomplishing what you were asked to do?",
                "Effort: How hard did you have to work to accomplish your level of performance?",
                "Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?",
                "Attentiveness: How focused were you on performing the task?"
            ]
            labels = [
                ["extremely alert", "", "alert", "", "neither alert nor sleepy", "", "sleepy", "", "very sleepy"],
                ["very low", "", "", "medium", "", "", "very high"],
                #["very low", "", "", "medium", "", "", "very high"],
                ["very slow", "", "", "medium", "", "", "very fast"],
                ["failure", "", "", "okay", "", "", "perfect"],
                ["very low", "", "", "medium", "", "", "very high"],
                ["very low", "", "", "medium", "", "", "very high"],
                ["not at all", "", "", "", "", "", "", "completely"],
            ]
            ticks = [
                (1, 2, 3, 4, 5, 6, 7, 8, 9),
                (1, 2, 3, 4, 5, 6, 7),
                #(1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7)
            ]
            for hint, label, tick in zip(
                texts, labels, ticks
            ):
                hint_text = visual.TextBox2(
                     win, text=hint, font='Open Sans',
                     pos=(0, 0.2), units='norm', letterHeight=self.FONT_SIZE / 1.5,
                     size=(self.LABEL_WIDTH, self.SLIDER_SIZE), borderWidth=0.0,
                     color='black', colorSpace='rgb',
                     opacity=None, bold=False, italic=False,
                     lineSpacing=1.0, speechPoint=None,
                     padding=0.0, alignment='center',
                     anchor='top-center', overflow='visible',
                     fillColor=None, borderColor=None,
                     flipHoriz=False, flipVert=False, languageStyle='LTR',
                     editable=False,
                     name=hint.lower().replace(" ", "_").replace(",", "_").replace("?", ""), 
                     autoLog=False
                )
                hint_text.setAutoDraw(True)
                self.COMPONENTS.append(hint_text)
                self.HINTS.append(hint_text)
                slider = visual.Slider(win=win, name=hint.lower().replace(" ", "_").replace(",", "_").replace("?", "_slider"),
                    startValue=None, size=(self.SLIDER_WIDTH, self.SLIDER_SIZE), pos=(0, 0.2 - self.SLIDER_SIZE * 2), units='norm',
                    labels=label, 
                    ticks=tick, 
                    granularity=1,
                    style='rating', styleTweaks=('triangleMarker',), opacity=None,
                    labelColor='black', markerColor='Red', lineColor='darkgrey', colorSpace='rgb',
                    font='Open Sans', labelHeight=self.FONT_SIZE / 3,
                    flip=False, ori=0.0, readOnly=False)
                slider.setAutoDraw(True)
                self.COMPONENTS.append(slider)
                self.SLIDERS.append(slider)
    
    
    class FormOverlayDeprecated(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        SEARCHBAR_SIZE = 0.1
        SLIDER_SIZE = 0.06
        SLIDER_WIDTH = 1.6
        LABEL_WIDTH = 2.0
        ASPECT_RATIO = win.size[0] / win.size[1]
        SLIDERS = []
        HINTS = []
        
        
        def __init__(self, random=False):
            super().__init__()
            self.random = random
            self.draw_interface()
            
        def update(self):
            for slider_component in self.SLIDERS:
                if slider_component.getRating() is None:
                    return True
            return False
            
        def log(self):
            for slider_component in self.SLIDERS:
                thisExp.addData(f"{slider_component.name}.rating", slider_component.getRating())
                thisExp.addData(f"{slider_component.name}.rt", slider_component.getRT())
            
        def reset(self):
            for slider_component in self.SLIDERS:
                slider_component.reset()
            if self.random:
                upper_bound = 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE * 2
                lower_bound = -1.0 + self.BAR_SIZE
                locations = np.linspace(upper_bound, lower_bound, len(self.SLIDERS) + 1)[:len(self.SLIDERS)]
                if self.random:
                    locations = np.random.permutation(locations)
                for hint, slider, location in zip(self.HINTS, self.SLIDERS, locations):
                    hint.pos = (0, location)
                    slider.pos = (0, location - self.SLIDER_SIZE * 2)
        
        def draw_interface(self):
            upper_bound = 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE * 2
            lower_bound = -1.0 + self.BAR_SIZE
            texts = [
                "Sleepiness: How sleepy are you?",
                "Mental Demand: How mentally demanding was the task?",
                "Physical Demand: How physically demanding was the task?",
                "Temporal Demand: How hurried or rushed was the pace of the task?",
                "Performance: How successful were you in accomplishing what you were asked to do?",
                "Effort: How hard did you have to work to accomplish your level of performance?",
                "Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?"
            ]
            labels = [
                ["extremely alert", "", "alert", "", "neither alert nor sleepy", "", "sleepy", "", "very sleepy"],
                ["very low", "", "", "medium", "", "", "very high"],
                ["very low", "", "", "medium", "", "", "very high"],
                ["very slow", "", "", "medium", "", "", "very fast"],
                ["failure", "", "", "okay", "", "", "perfect"],
                ["very low", "", "", "medium", "", "", "very high"],
                ["very low", "", "", "medium", "", "", "very high"],
            ]
            ticks = [
                (1, 2, 3, 4, 5, 6, 7, 8, 9),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7),
                (1, 2, 3, 4, 5, 6, 7)
            ]
            locations = np.linspace(upper_bound, lower_bound, len(texts) + 1)[:len(texts)]
            if self.random:
                locations = np.random.permutation(locations)
            for hint, label, tick, location in zip(
                texts, labels, ticks, locations
                
            ):
                hint_text = visual.TextBox2(
                     win, text=hint, font='Open Sans',
                     pos=(0, location), units='norm', letterHeight=self.FONT_SIZE / 2,
                     size=(self.LABEL_WIDTH, self.SLIDER_SIZE), borderWidth=0.0,
                     color='black', colorSpace='rgb',
                     opacity=None, bold=False, italic=False,
                     lineSpacing=1.0, speechPoint=None,
                     padding=0.0, alignment='center',
                     anchor='top-center', overflow='visible',
                     fillColor=None, borderColor=None,
                     flipHoriz=False, flipVert=False, languageStyle='LTR',
                     editable=False,
                     name=hint.lower().replace(" ", "_").replace("?", ""), 
                     autoLog=False
                )
                hint_text.setAutoDraw(True)
                self.COMPONENTS.append(hint_text)
                self.HINTS.append(hint_text)
                slider = visual.Slider(win=win, name=hint.lower().replace(" ", "_").replace("?", "_slider"),
                    startValue=None, size=(self.SLIDER_WIDTH, self.SLIDER_SIZE), pos=(0, location - self.SLIDER_SIZE * 2), units='norm',
                    labels=label, 
                    ticks=tick, 
                    granularity=1,
                    style='rating', styleTweaks=('triangleMarker',), opacity=None,
                    labelColor='black', markerColor='Red', lineColor='darkgrey', colorSpace='rgb',
                    font='Open Sans', labelHeight=self.FONT_SIZE / 4,
                    flip=False, ori=0.0, readOnly=False)
                slider.setAutoDraw(True)
                self.COMPONENTS.append(slider)
                self.SLIDERS.append(slider)
    
    
    class BrowserSearchbar(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        SEARCHBAR_SIZE = 0.1
        ASPECT_RATIO = win.size[0] / win.size[1]
        WORD_LIST = ['backspace', 'comma', 'period', 'underscore', 'minus', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        CHAR_LIST = [',', '.', '_', '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        
        
        def __init__(self):
            super().__init__()
            self.draw_interface()
            self.allocate_target()
            
        def log(self):
            thisExp.addData('browser_navigation_template', self.target_name)
            thisExp.addData('browser_navigation_user_input', self.get_text())
        
        def update(self):
            self.search_bar.text = "".join([c for c in self.search_bar.text if c in self.CHAR_LIST])
            if self.search_bar.text.replace(" ", "").lower() == self.target_name.lower():
                self.search_bar.editable = False
                self.search_bar.hasFocus = False
                return False
            self.search_bar.editable = True
            self.search_bar.hasFocus = True
            return True
        
        def get_text(self):
            return self.search_bar.text
            
        def allocate_target(self, url=None):
            if url is None:
                self.target_name = fake.url().replace("http://", "").replace("https://", "").strip("/") # np.random.choice(["www.google.com", "www.bing.com"])
            else:
                self.target_name = url
            self.search_bar.reset()
        
        def draw_interface(self):
            left_circle = visual.ShapeStim(
                win=win, name='base_searchbar_left_circle',units='norm', 
                size=(self.SEARCHBAR_SIZE / self.ASPECT_RATIO, self.SEARCHBAR_SIZE), 
                vertices='circle',
                ori=0.0, 
                pos=(-1.0 + self.SEARCHBAR_SIZE / self.ASPECT_RATIO, 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE / 2), 
                anchor='top-left',
                lineWidth=1.0,
                colorSpace='rgb', lineColor=None, fillColor='lightgray',
                opacity=1.0, interpolate=True)
            left_circle.setAutoDraw(True)
            self.COMPONENTS.append(left_circle)
            right_circle = visual.ShapeStim(
                win=win, name='base_searchbar_right_circle',units='norm', 
                size=(self.SEARCHBAR_SIZE / self.ASPECT_RATIO, self.SEARCHBAR_SIZE), 
                vertices='circle',
                ori=0.0, 
                pos=(1.0 - self.SEARCHBAR_SIZE / self.ASPECT_RATIO, 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE / 2), 
                anchor='top-right',
                lineWidth=1.0,
                colorSpace='rgb', lineColor=None, fillColor='lightgray',
                opacity=1.0, interpolate=True)
            right_circle.setAutoDraw(True)
            self.COMPONENTS.append(right_circle)
            bar = visual.Rect(
                win=win, name='base_searchbar_bar',units='norm', 
                size=(2.0 - 3 * self.SEARCHBAR_SIZE / self.ASPECT_RATIO, self.SEARCHBAR_SIZE), 
                ori=0.0, 
                pos=(0.0, 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE / 2), 
                anchor='top-center',
                lineWidth=1.0,
                colorSpace='rgb', lineColor=None, fillColor='lightgray',
                opacity=1.0, interpolate=True)
            bar.setAutoDraw(True)
            self.COMPONENTS.append(bar)
            self.search_bar = visual.TextBox2(
                win=win,
                name="base_searchbar_input", 
                placeholder="",
                text="",
                font="Open Sans",
                letterHeight=self.FONT_SIZE / 2,
                alignment="center-left",
                units='norm',
                anchor='top-center',
                padding=0.01,
                lineSpacing=1.0,
                bold=False,
                italic=False,
                color="black",
                borderWidth=0.0,
                fillColor=None,
                borderColor=None,
                overflow="visible",
                editable=False,
                pos=(0, 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.SEARCHBAR_SIZE / 2),
                size=(2.0 - 3 * self.SEARCHBAR_SIZE / self.ASPECT_RATIO, self.SEARCHBAR_SIZE))
            self.search_bar.setAutoDraw(True)
            self.COMPONENTS.append(self.search_bar)
    
    
    class TrashItemsOverlay(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        FOLDER_SIZE = 0.1
        ASPECT_RATIO = win.size[0] / win.size[1]
        FOLDERS = []
        FOLDER_HIGHLIGHTS = []
        
        TARGET = 'resources/trash_stimuli.png'
        TARGET_HIGHLIGHT = 'resources/trash_stimuli_highlight.png'
        
        mouse = event.Mouse(win=win)
        last_mouse_pressed = False
        last_mouse_location = (0.0, 0.0)
        
        def __init__(self, N=6):
            super().__init__()
            self.N = N
            self.draw_interface()
            
        def highlight_folders(self, pos):
            count = 0
            for folder_highlight in self.FOLDER_HIGHLIGHTS:
                corners = [
                    (folder_highlight.pos[0] + x, folder_highlight.pos[1] + y)
                    for x in [folder_highlight.size[0] / -2, folder_highlight.size[0] / 2] for y in [folder_highlight.size[1] / -2, folder_highlight.size[1] / 2]
                ]
                found = False
                for corner in corners:
                    if min(self.last_mouse_location[0], pos[0]) <= corner[0] <= max(self.last_mouse_location[0], pos[0]) and \
                        min(self.last_mouse_location[1], pos[1]) <= corner[1] <= max(self.last_mouse_location[1], pos[1]):
                            count += 1
                            found = True
                            folder_highlight.opacity = 1.0
                            break
                if not found:
                    folder_highlight.opacity = 0.0
            return not count == len(self.FOLDER_HIGHLIGHTS)
        
        def highlight_area(self, pos, show=False):
            if show:
                self.area.opacity = 0.8
                self.area.pos = (min(pos[0], self.last_mouse_location[0]) + abs(pos[0] - self.last_mouse_location[0]) / 2, 
                                    min(pos[1], self.last_mouse_location[1]) + abs(pos[1] - self.last_mouse_location[1]) / 2)
                self.area.size = (abs(pos[0] - self.last_mouse_location[0]), abs(pos[1] - self.last_mouse_location[1]))
            else:
                self.area.opacity = 0.0
            
        def update(self):
            buttons = self.mouse.getPressed()
            if not self.last_mouse_pressed and np.sum(buttons) > 0:
                self.last_mouse_pressed = True
                self.last_mouse_location = self.mouse.getPos()
                self.highlight_area(self.mouse.getPos(), show=True)
                # Highlight all selected folders
                self.highlight_folders(self.mouse.getPos())
                return True
            elif self.last_mouse_pressed and np.sum(buttons) > 0:
                self.highlight_area(self.mouse.getPos(), show=True)
                # Highlight all selected folders
                self.highlight_folders(self.mouse.getPos())
                return True
            elif self.last_mouse_pressed and np.sum(buttons) == 0:
                self.highlight_area(self.mouse.getPos(), show=False)
                # Highlight all selected folders
                self.last_mouse_pressed = False
                return self.highlight_folders(self.mouse.getPos())
            self.highlight_area(self.mouse.getPos(), show=False)
            # None selected
            return True
            
        def log(self):
            for folder in self.FOLDERS:
                thisExp.addData(folder.name, folder.pos)
            thisExp.addData("selection_start", self.last_mouse_location)
            thisExp.addData("selection_end", self.mouse.getPos())
            
        def reset(self):
            self.last_mouse_pressed = False
            upper_bound = 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE * 2
            lower_bound = -1.0 + self.BAR_SIZE + self.TOOLBAR_SIZE
            left_bound = -1.0 + self.TOOLBAR_SIZE / self.ASPECT_RATIO + self.FOLDER_SIZE / 2 / self.ASPECT_RATIO
            right_bound = 1.0 - self.TOOLBAR_SIZE / self.ASPECT_RATIO - self.FOLDER_SIZE / 2 / self.ASPECT_RATIO
            locs_x = np.random.uniform(low=left_bound, high=right_bound, size=self.N)
            locs_y = np.random.uniform(low=lower_bound, high=upper_bound, size=self.N)
            self.area.opacity = 0.0
            self.area.size = (0.0, 0.0)
            self.area.pos = (0.0, 0.0)
            for folder, folder_highlight, loc_x, loc_y in zip(self.FOLDERS, self.FOLDER_HIGHLIGHTS, locs_x, locs_y):
                folder.pos = (loc_x, loc_y)
                folder_highlight.pos = (loc_x, loc_y)
                folder_highlight.opacity = 0.0
        
        def draw_interface(self):
            self.area = visual.Rect(
                win=win, name='selection_area',units='norm', 
                size=(0.0, 0.0), 
                ori=0.0, 
                pos=(0.0, 0.0), 
                anchor='center',
                lineWidth=2.0,
                colorSpace='rgb',
                lineColor=[-1.0000, -0.0588, 0.6863],
                fillColor=[0.3961, 0.6314, 1.0000],
                opacity=0.0, interpolate=True)
            self.area.setAutoDraw(True)
            self.COMPONENTS.append(self.area)
            upper_bound = 1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE * 2
            lower_bound = -1.0 + self.BAR_SIZE + self.TOOLBAR_SIZE
            left_bound = -1.0 + self.TOOLBAR_SIZE / self.ASPECT_RATIO + self.FOLDER_SIZE / 2 / self.ASPECT_RATIO
            right_bound = 1.0 - self.TOOLBAR_SIZE / self.ASPECT_RATIO - self.FOLDER_SIZE / 2 / self.ASPECT_RATIO
            locs_x = np.random.uniform(low=left_bound, high=right_bound, size=self.N)
            locs_y = np.random.uniform(low=lower_bound, high=upper_bound, size=self.N)
            for index, (loc_x, loc_y) in enumerate(zip(
                locs_x, locs_y
            )):
                folder_highlight = visual.ImageStim(
                    win=win,
                    name=f"trash_folder_highlight_{index}", 
                    units='norm', 
                    opacity=0.0,
                    image=self.TARGET_HIGHLIGHT, anchor='center',
                    pos=(loc_x, loc_y), size=(self.FOLDER_SIZE / self.ASPECT_RATIO, self.FOLDER_SIZE),
                    interpolate=True)
                folder_highlight.setAutoDraw(True)
                self.COMPONENTS.append(folder_highlight)
                self.FOLDER_HIGHLIGHTS.append(folder_highlight)
                folder = visual.ImageStim(
                    win=win,
                    name=f"trash_folder_{index}", 
                    units='norm', 
                    image=self.TARGET, anchor='center',
                    pos=(loc_x, loc_y), size=(self.FOLDER_SIZE / self.ASPECT_RATIO, self.FOLDER_SIZE),
                    interpolate=True)
                folder.setAutoDraw(True)
                self.COMPONENTS.append(folder)
                self.FOLDERS.append(folder)
                
    class PopupWindow(Component):
        ICON = 'resources/warning.png'
        WINDOW_SIZE = 0.5
        TOOLBAR_SIZE = 0.1
        FONT_SIZE = 0.1
        BUTTON_SIZE = 0.1
        ASPECT_RATIO = win.size[0] / win.size[1]
        
        OPTIONS = {
            "mac": ["Yes", "No"],
            "windows": ["No", "Yes"]
        }
        target_name = "Yes"
        
        mouse = event.Mouse(win=win)
        BUTTONS = []
        BUTTON_TEXTS = []
        
        def __init__(self, style):
            super().__init__()
            self.style = style
            self.draw_interface()
            
        def update(self):
            for element, element_text in zip(self.BUTTONS, self.BUTTON_TEXTS):
                if element.contains(self.mouse.getPos()):
                    if element_text.text == self.target_name:
                        element.fillColor = [0.7569, 0.8667, 0.9529]
                    else:
                        element.fillColor = [0.6, 0.6, 0.6]
                else:
                    element.fillColor = [0.9843, 0.9843, 0.9843]
                if self.mouse.isPressedIn(element) and element_text.text == self.target_name:
                    return False
            return True
            
        def reset(self):
            style = self.style.current_style
            for button_text, override in zip(self.BUTTON_TEXTS, self.OPTIONS[style]):
                button_text.text = override
            
        def draw_interface(self):
            style = self.style.current_style
            bg_window = visual.Rect(
                win=win,
                name="base_popup", 
                units='norm',
                fillColor=[0.7804, 0.7804, 0.7804], opacity=0.96, anchor='center',
                pos=(0, 0), size=(2 * self.WINDOW_SIZE, self.WINDOW_SIZE),
                interpolate=True)
            bg_window.setAutoDraw(True)
            self.COMPONENTS.append(bg_window)
            bg_toolbar = visual.Rect(
                win=win,
                name="base_popup_toolbar", 
                units='norm',
                fillColor="lightgray", opacity=1.0, anchor='top-center',
                pos=(0, self.WINDOW_SIZE / 2), size=(2 * self.WINDOW_SIZE, self.TOOLBAR_SIZE),
                interpolate=True)
            bg_toolbar.setAutoDraw(True)
            self.COMPONENTS.append(bg_toolbar)
            icon_size = self.WINDOW_SIZE - self.TOOLBAR_SIZE * 2 - self.BUTTON_SIZE
            icon = visual.ImageStim(
                win=win,
                name="base_popup_icon",
                units="norm",
                image=self.ICON,
                opacity=1.0,
                size=(icon_size / self.ASPECT_RATIO, icon_size),
                pos=(-1 * self.WINDOW_SIZE + self.TOOLBAR_SIZE / 2, self.TOOLBAR_SIZE / -2),
                anchor="center-left"
            )
            icon.setAutoDraw(True)
            self.COMPONENTS.append(icon)
            
            content = visual.TextBox2(
                 win, text="Are you sure you want to permanently delete these files?", font='Open Sans',
                 pos=(-1 * self.WINDOW_SIZE + icon_size, self.TOOLBAR_SIZE / -2), units='norm', letterHeight=self.FONT_SIZE / 2,
                 size=(2 * self.WINDOW_SIZE - icon_size - self.TOOLBAR_SIZE, icon_size), borderWidth=2.0,
                 color='black', colorSpace='rgb',
                 opacity=None, bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.0, alignment='top-left',
                 anchor='bottom-left', overflow='visible',
                 fillColor=None, borderColor=None,
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='base_popup_content', 
                 autoLog=False
            )
            content.setAutoDraw(True)
            self.COMPONENTS.append(content)
            
            for index, button_name in enumerate(self.OPTIONS[style]):
                button = visual.Rect(
                    win=win,
                    name=f"base_popup_button_{index}", 
                    units='norm',
                    lineColor=[-1.0000, -0.0588, 0.6863], lineWidth=1.0,
                    fillColor=[0.9843, 0.9843, 0.9843], opacity=1.0, anchor='bottom-right',
                    pos=(self.WINDOW_SIZE - self.BUTTON_SIZE * (index + 1) / 2 - self.BUTTON_SIZE * index * 2, self.WINDOW_SIZE / -2 + self.TOOLBAR_SIZE / 2),
                    size=(self.BUTTON_SIZE * 2, self.BUTTON_SIZE),
                    interpolate=True)
                # Highlight [0.7569, 0.8667, 0.9529]
                button.setAutoDraw(True)
                self.COMPONENTS.append(button)
                self.BUTTONS.append(button)
                button_content = visual.TextBox2(
                     win, text=button_name, font='Open Sans',
                     pos=(self.WINDOW_SIZE - self.BUTTON_SIZE * (index + 1) / 2 - self.BUTTON_SIZE * index * 2, self.WINDOW_SIZE / -2 + self.TOOLBAR_SIZE / 2),
                     size=(self.BUTTON_SIZE * 2, self.BUTTON_SIZE),
                     units='norm', letterHeight=self.FONT_SIZE / 2,
                     borderWidth=2.0,
                     color='black', colorSpace='rgb',
                     opacity=None, bold=False, italic=False,
                     lineSpacing=1.0, speechPoint=None,
                     padding=0.0, alignment='center',
                     anchor='bottom-right', overflow='visible',
                     fillColor=None, borderColor=None,
                     flipHoriz=False, flipVert=False, languageStyle='LTR',
                     editable=False,
                     name=f"base_popup_button_text_{index}", 
                     autoLog=False
                )
                button_content.setAutoDraw(True)
                self.COMPONENTS.append(button_content)
                self.BUTTON_TEXTS.append(button_content)
    
    class SplitViewEditor(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        TOOLBAR_SIZE = 0.1
        PADDING_SIZE = 0.05
        
        ASPECT_RATIO = win.size[0] / win.size[1]
        
        TARGET = ""
        
        def __init__(self):
            super().__init__()
            self.draw_interface()
            self.allocate_target()
            
        def allocate_target(self):
            self.TARGET = fake.sentence(nb_words=int(personalized_settings["typing_speed"] / 4.7 * 45)+1)   #"TODO: Add template"
            self.reset()
        
        def allocate_test_target(self):
            self.TARGET = fake.sentence(nb_words=40)
            self.reset()
            
        def log(self):
            thisExp.addData("notes_repeat_source", self.notes_repeat_source.text)
            thisExp.addData("notes_repeat_target", self.notes_repeat_target.text)
        
        def safe_match(self, input_string):
            return len(re.sub(r'[^A-Za-z,. \n]', '', input_string))
        
        def update(self):
            matched = self.safe_match(self.notes_repeat_source.text) <= self.safe_match(self.notes_repeat_target.text)
            if matched:
                self.notes_repeat_target.hasFocus = False
                self.notes_repeat_target.editable = False
                return False
            self.notes_repeat_target.hasFocus = True
            self.notes_repeat_target.editable = True
            return True
            
        def reset(self):
            self.notes_repeat_target.clear()
            self.notes_repeat_source.reset()
            self.notes_repeat_target.reset()
            self.notes_repeat_source.text = self.TARGET
            
        def draw_interface(self):
            self.notes_repeat_source = visual.TextBox2(
                 win, text='This is the template.\nMultiline.', placeholder='Type here...', font='Open Sans',
                 pos=(-1 * self.PADDING_SIZE / 2, (1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.PADDING_SIZE) / 2),
                 size=((1.0 - 2 * self.PADDING_SIZE) / 2 * self.ASPECT_RATIO, (2.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.BAR_SIZE - 2 * self.PADDING_SIZE) / 2),
                 borderWidth=2.0, 
                 units='height',
                 letterHeight=self.FONT_SIZE / 2 / 2,
                 color='black', colorSpace='rgb',
                 opacity=None,
                 bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.05, alignment='top-left',
                 anchor='top-right', overflow='visible',
                 fillColor='white', borderColor='black',
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='notes_repeat_source',
                 autoLog=False,
                 autoDraw=True
            )
            self.COMPONENTS.append(self.notes_repeat_source)
            self.notes_repeat_target = visual.TextBox2(
                 win, text=None, placeholder='Type here...', font='Open Sans',
                 pos=(self.PADDING_SIZE / 2, (1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.PADDING_SIZE) / 2),
                 size=((1.0 - 2 * self.PADDING_SIZE) / 2 * self.ASPECT_RATIO, (2.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.BAR_SIZE - 2 * self.PADDING_SIZE) / 2),
                 units='height',
                 letterHeight=self.FONT_SIZE / 2 / 2,
                 borderWidth=2.0,
                 color='black', colorSpace='rgb',
                 opacity=None,
                 bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.05, alignment='top-left',
                 anchor='top-left', overflow='visible',
                 fillColor='white', borderColor='black',
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='notes_repeat_target',
                 autoLog=False,
                 autoDraw=True
            )
            self.COMPONENTS.append(self.notes_repeat_target)
            
    
    class SingleViewEditor(SplitViewEditor):
        def log(self):
            thisExp.addData("single_note_repeat_source", self.notes_repeat_source.text)
            thisExp.addData("single_note_repeat_target", self.notes_repeat_target.text)
        
        def draw_interface(self):
            self.notes_repeat_source = visual.TextBox2(
                 win, text='This is the template.\nMultiline.', placeholder='Type here...', font='Open Sans',
                 pos=(0, (1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.PADDING_SIZE) / 2),
                 size=((1.0 - 2 * self.PADDING_SIZE) * self.ASPECT_RATIO, (2.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.BAR_SIZE - 2 * self.PADDING_SIZE) / 2),
                 borderWidth=2.0, 
                 units='height',
                 letterHeight=self.FONT_SIZE / 2 / 2,
                 color='gray', colorSpace='rgb',
                 opacity=None,
                 bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.05, alignment='top-left',
                 anchor='top-center', overflow='visible',
                 fillColor='white', borderColor='black',
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='single_note_repeat_source',
                 autoLog=False,
                 autoDraw=True
            )
            self.COMPONENTS.append(self.notes_repeat_source)
            self.notes_repeat_target = visual.TextBox2(
                 win, text=None, placeholder='', font='Open Sans',
                 pos=(0, (1.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.PADDING_SIZE) / 2),
                 size=((1.0 - 2 * self.PADDING_SIZE) * self.ASPECT_RATIO, (2.0 - self.FONT_SIZE - self.TOOLBAR_SIZE - self.BAR_SIZE - 2 * self.PADDING_SIZE) / 2),
                 units='height',
                 letterHeight=self.FONT_SIZE / 2 / 2,
                 borderWidth=2.0,
                 color='black', colorSpace='rgb',
                 opacity=0.1,
                 bold=False, italic=False,
                 lineSpacing=1.0, speechPoint=None,
                 padding=0.05, alignment='top-left',
                 anchor='top-center', overflow='visible',
                 fillColor='white', borderColor='black',
                 flipHoriz=False, flipVert=False, languageStyle='LTR',
                 editable=False,
                 name='single_note_repeat_target',
                 autoLog=False,
                 autoDraw=True
            )
            self.COMPONENTS.append(self.notes_repeat_target)
    
    
    class Notification(Component):
        FONT_SIZE = 0.1
        BAR_SIZE = 0.1
        WIDTH = 0.4
        HEIGHT = 0.2
        
        ASPECT_RATIO = win.size[0] / win.size[1]
        
        LOCATIONS = {
            "windows": {
                "anchor": "bottom-right",
                "image": "resources/windows/notification.png",
                "size": (WIDTH, HEIGHT),
                "pos": (0.5 * ASPECT_RATIO, -0.5 + BAR_SIZE / 2)
            },
            "mac": {
                "anchor": "top-right",
                "image": "resources/mac/notification.png",
                "size": (WIDTH, HEIGHT),
                "pos": (0.5 * ASPECT_RATIO, 0.5 - FONT_SIZE / 2)
            }
        }
        
        mouse = event.Mouse(win=win)
        
        def __init__(self, style):
            super().__init__()
            self.style = style
            self.draw_interface()
        
        def update(self):
            if self.mouse.isPressedIn(self.background):
                return False
            return True
            
        def reset(self):
            style = self.style.current_style
            for key, value in self.LOCATIONS[style].items():
                setattr(self.background, key, value)
            
        def draw_interface(self):
            style = self.style.current_style
            self.background = visual.ImageStim(
                win=win,
                name="notification_background", 
                units='height', 
                interpolate=True,
                **self.LOCATIONS[style])
            self.background.setAutoDraw(True)
            self.COMPONENTS.append(self.background)
    
    
    
    class CameraView(Component):
        BAR_SIZE = 0.1
        
        def __init__(self):
            super().__init__()
            self.draw_interface()
            
        def draw_interface(self):
            self.view = visual.ImageStim(
                win=win,
                name='live_camera_view', 
                image=camera_connector, mask=None, anchor='top-left',
                ori=0.0, pos=(-1, 1), draggable=False, size=(self.BAR_SIZE, self.BAR_SIZE),
                color=[1,1,1], colorSpace='rgb', opacity=None,
                flipHoriz=False, flipVert=False,
                texRes=128.0, interpolate=True, depth=-1.0)
            self.view.setAutoDraw(True)
            self.COMPONENTS.append(self.view)
        
    
    def hide_all():
        task_bar.hide()
        base_window.hide()
        browser_searchbar.hide()
        form_overlay.hide()
        trash_items_overlay.hide()
        popup_window.hide()
        split_view_editor.hide()
        single_view_editor.hide()
        notification.hide()
        
    def reset_layout():
        stylizer.reset()
        task_bar.reset()
        base_window.reset()
        popup_window.reset()
        split_view_editor.reset()
        single_view_editor.reset()
        notification.reset()
    
    class EventTimer():
        showed_alertness_prompt = 0
        interval_alertness_prompt = 5 * 60  # seconds
        duration_experiment = 2 * 60 * 60 + 5 * 60   # seconds
        keywords = ["MAIL", "FILE_DRAGGING", "FILE_OPENING", "TRASH_BIN", "NOTES"]
        
        def __init__(self):
            self.clock = core.MonotonicClock()
        
        def should_end(self):
            if self.clock.getTime() > self.duration_experiment:
                return True
            return False
        
        def should_show(self, target):
            if target not in self.keywords:
                expected_prompt_counts = int(self.clock.getTime() / self.interval_alertness_prompt)
                if self.showed_alertness_prompt < expected_prompt_counts:
                    self.showed_alertness_prompt += 1
                    return 1
                return 0
            if tasks.thisN == self.keywords.index(target):
                return 1
            return 0
    
    stylizer = Stylizer()
    task_bar = TaskBar(style=stylizer)
    base_window = BaseWindow(style=stylizer)
    browser_searchbar = BrowserSearchbar()
    form_overlay = FormOverlay(random=True)
    trash_items_overlay = TrashItemsOverlay(N=8)
    popup_window = PopupWindow(style=stylizer)
    split_view_editor = SplitViewEditor()
    single_view_editor = SingleViewEditor()
    notification = Notification(style=stylizer)
    camera_view = CameraView()
    
    hide_all()
    
    global_text = visual.TextStim(win=win, name='global_text',
        text='Loading...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "calibration_start" ---
    calibration_start_text = visual.TextStim(win=win, name='calibration_start_text',
        text='Calibration Stage\n\n\nWe will run through a few tests to ensure you the best experience.\n\npress any key to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    calibration_start_mouse = event.Mouse(win=win)
    x, y = [None, None]
    calibration_start_mouse.mouseClock = core.Clock()
    calibration_start_key = keyboard.Keyboard(deviceName='calibration_start_key')
    
    # --- Initialize components for Routine "calibration_appearance_windows" ---
    calibration_appearance_windows_text = visual.TextStim(win=win, name='calibration_appearance_windows_text',
        text='Introduction to System Appearance\n\n\nWe will randomize system appearance throughout the experiment. With Windows style layout, title bar buttons are presented on the top-right corner of the window. On the task bar, we have applications named "File Manager", "Browser", "Mail", "Document" and "Trash", in the order of their appearance from left to right.\n\nclick CLOSE button on the title bar to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    calibration_appearance_windows_prompt = visual.TextBox2(
         win, text='Guidance will be shown HERE', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='calibration_appearance_windows_prompt',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "calibration_appearance_mac" ---
    calibration_appearance_mac_text = visual.TextStim(win=win, name='calibration_appearance_mac_text',
        text='Introduction to System Appearance\n\n\nWith macOS style layout, title bar buttons are presented on the top-left corner of the window. On the task bar, we have applications named "File Manager", "Browser", "Mail", "Document" and "Trash", in the order of their appearance from left to right.\n\nclick CLOSE button on the title bar to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    calibration_appearance_mac_prompt = visual.TextBox2(
         win, text='Guidance will be shown HERE', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='calibration_appearance_mac_prompt',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "calibration_typing_start" ---
    calibration_typing_start_text = visual.TextStim(win=win, name='calibration_typing_start_text',
        text='Typing Calibration\n\n\nYou will be asked to enter sentences consisting of random words, with a period "." in the end. Your mouse will be disabled for this task. During typing, if you find any previous errors in typing that happened for more than two words before, you should ignore it and continue typing instead of trying to fix it.\n\npress ENTER to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    calibration_typing_start_key = keyboard.Keyboard(deviceName='calibration_typing_start_key')
    
    # --- Initialize components for Routine "calibration_typing" ---
    calibration_typing_key = keyboard.Keyboard(deviceName='calibration_typing_key')
    
    # --- Initialize components for Routine "calibration_end" ---
    calibration_end_text = visual.TextStim(win=win, name='calibration_end_text',
        text='End of Calibration\n\n\nYour personalized settings have been applied.\n\npress any key to continue...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    calibration_end_mouse = event.Mouse(win=win)
    x, y = [None, None]
    calibration_end_mouse.mouseClock = core.Clock()
    calibration_end_key = keyboard.Keyboard(deviceName='calibration_end_key')
    calibration_end_debug_text = visual.TextStim(win=win, name='calibration_end_debug_text',
        text='',
        font='Open Sans',
        pos=(0, 0.5), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='gray', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "experiment_start" ---
    experiment_start_text = visual.TextStim(win=win, name='experiment_start_text',
        text='Start of Experiment\n\n\nWelcome to this exciting journey, please try to finish all the tasks as fast as you can!\n\npress any key to start...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    experiment_start_mouse = event.Mouse(win=win)
    x, y = [None, None]
    experiment_start_mouse.mouseClock = core.Clock()
    experiment_start_key = keyboard.Keyboard(deviceName='experiment_start_key')
    
    # --- Initialize components for Routine "style_randomizer" ---
    
    # --- Initialize components for Routine "mail_homescreen" ---
    mail_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    mail_mouse_homescreen.mouseClock = core.Clock()
    mail_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='mail_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "mail_notification" ---
    mail_notification_mouse = event.Mouse(win=win)
    x, y = [None, None]
    mail_notification_mouse.mouseClock = core.Clock()
    mail_notification_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='mail_notification_textbox',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "mail_content" ---
    mail_content_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='mail_content_textbox',
         depth=-1, autoLog=False,
    )
    mail_content_user_key_release = keyboard.Keyboard(deviceName='mail_content_user_key_release')
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "file_manager_homescreen" ---
    file_manager_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    file_manager_mouse_homescreen.mouseClock = core.Clock()
    file_manager_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='file_manager_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "file_manager_dragging" ---
    file_manager_dragging_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='file_manager_dragging_textbox',
         depth=-1, autoLog=False,
    )
    stimuli_dragging_mouse = event.Mouse(win=win)
    x, y = [None, None]
    stimuli_dragging_mouse.mouseClock = core.Clock()
    stimuli_dragging_target_image = visual.ImageStim(
        win=win,
        name='stimuli_dragging_target_image', units='norm', 
        image='resources/target_stimuli.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=0.8,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    stimuli_dragging_stimuli_image = visual.ImageStim(
        win=win,
        name='stimuli_dragging_stimuli_image', units='norm', 
        image='resources/drag_stimuli.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=0.9,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "file_manager_homescreen" ---
    file_manager_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    file_manager_mouse_homescreen.mouseClock = core.Clock()
    file_manager_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='file_manager_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "file_manager_opening" ---
    file_manager_opening_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='file_manager_opening_textbox',
         depth=-1, autoLog=False,
    )
    stimuli_reaction_mouse_movement = event.Mouse(win=win)
    x, y = [None, None]
    stimuli_reaction_mouse_movement.mouseClock = core.Clock()
    stimuli_reaction_stimuli_image = visual.ImageStim(
        win=win,
        name='stimuli_reaction_stimuli_image', units='norm', 
        image='resources/folder_stimuli.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    stimuli_reaction_mouse_debug = visual.TextStim(win=win, name='stimuli_reaction_mouse_debug',
        text='',
        font='Open Sans',
        units='norm', pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='lightgrey', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "trash_bin_homescreen" ---
    trash_bin_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    trash_bin_mouse_homescreen.mouseClock = core.Clock()
    trash_bin_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trash_bin_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "trash_bin_select" ---
    trash_bin_select_mouse = event.Mouse(win=win)
    x, y = [None, None]
    trash_bin_select_mouse.mouseClock = core.Clock()
    trash_bin_select_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trash_bin_select_textbox',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "trash_bin_confirm" ---
    trash_bin_confirm_mouse = event.Mouse(win=win)
    x, y = [None, None]
    trash_bin_confirm_mouse.mouseClock = core.Clock()
    trash_bin_confirm_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trash_bin_confirm_textbox',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "notes_homescreen" ---
    notes_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    notes_mouse_homescreen.mouseClock = core.Clock()
    notes_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='notes_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "notes_repeat" ---
    notes_repeat_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='notes_repeat_textbox',
         depth=-1, autoLog=False,
    )
    notes_repeat_keyboard = keyboard.Keyboard(deviceName='notes_repeat_keyboard')
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "browser_homescreen" ---
    browser_mouse_homescreen = event.Mouse(win=win)
    x, y = [None, None]
    browser_mouse_homescreen.mouseClock = core.Clock()
    browser_textbox_homescreen = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='browser_textbox_homescreen',
         depth=-2, autoLog=False,
    )
    
    # --- Initialize components for Routine "browser_navigation" ---
    browser_navigation_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='browser_navigation_textbox',
         depth=-1, autoLog=False,
    )
    browser_navigation_mouse = event.Mouse(win=win)
    x, y = [None, None]
    browser_navigation_mouse.mouseClock = core.Clock()
    browser_navigation_user_key_release = keyboard.Keyboard(deviceName='browser_navigation_user_key_release')
    
    # --- Initialize components for Routine "browser_content" ---
    browser_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='browser_textbox',
         depth=-1, autoLog=False,
    )
    browser_content_mouse = event.Mouse(win=win)
    x, y = [None, None]
    browser_content_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=[0,0], draggable=False, units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    window_close_mouse = event.Mouse(win=win)
    x, y = [None, None]
    window_close_mouse.mouseClock = core.Clock()
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "loop_end" ---
    
    # --- Initialize components for Routine "experiment_end" ---
    experiment_end_text = visual.TextStim(win=win, name='experiment_end_text',
        text='End of Experiment\n\n\nThank you very much for your participation!\n\npress any key to exit...',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    experiment_end_mouse = event.Mouse(win=win)
    x, y = [None, None]
    experiment_end_mouse.mouseClock = core.Clock()
    experiment_end_key = keyboard.Keyboard(deviceName='experiment_end_key')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "definition" ---
    # create an object to store info about Routine definition
    definition = data.Routine(
        name='definition',
        components=[global_text],
    )
    definition.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from global_code
    serial_connector.write(SerialConnector.EEG_START_RECORDING)
    
    # store start times for definition
    definition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    definition.tStart = globalClock.getTime(format='float')
    definition.status = STARTED
    definition.maxDuration = None
    # keep track of which components have finished
    definitionComponents = definition.components
    for thisComponent in definition.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "definition" ---
    definition.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *global_text* updates
        
        # if global_text is starting this frame...
        if global_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            global_text.frameNStart = frameN  # exact frame index
            global_text.tStart = t  # local t and not account for scr refresh
            global_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(global_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'global_text.started')
            # update status
            global_text.status = STARTED
            global_text.setAutoDraw(True)
        
        # if global_text is active this frame...
        if global_text.status == STARTED:
            # update params
            pass
        
        # if global_text is stopping this frame...
        if global_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > global_text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                global_text.tStop = t  # not accounting for scr refresh
                global_text.tStopRefresh = tThisFlipGlobal  # on global time
                global_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'global_text.stopped')
                # update status
                global_text.status = FINISHED
                global_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            definition.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in definition.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "definition" ---
    for thisComponent in definition.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for definition
    definition.tStop = globalClock.getTime(format='float')
    definition.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if definition.maxDurationReached:
        routineTimer.addTime(-definition.maxDuration)
    elif definition.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "calibration_start" ---
    # create an object to store info about Routine calibration_start
    calibration_start = data.Routine(
        name='calibration_start',
        components=[calibration_start_text, calibration_start_mouse, calibration_start_key],
    )
    calibration_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from calibration_start_code
    serial_connector.write(SerialConnector.CALIB_BEGIN)
    camera_connector.record()
    # setup some python lists for storing info about the calibration_start_mouse
    calibration_start_mouse.x = []
    calibration_start_mouse.y = []
    calibration_start_mouse.leftButton = []
    calibration_start_mouse.midButton = []
    calibration_start_mouse.rightButton = []
    calibration_start_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for calibration_start_key
    calibration_start_key.keys = []
    calibration_start_key.rt = []
    _calibration_start_key_allKeys = []
    # store start times for calibration_start
    calibration_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_start.tStart = globalClock.getTime(format='float')
    calibration_start.status = STARTED
    thisExp.addData('calibration_start.started', calibration_start.tStart)
    calibration_start.maxDuration = None
    # keep track of which components have finished
    calibration_startComponents = calibration_start.components
    for thisComponent in calibration_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_start" ---
    calibration_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *calibration_start_text* updates
        
        # if calibration_start_text is starting this frame...
        if calibration_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_start_text.frameNStart = frameN  # exact frame index
            calibration_start_text.tStart = t  # local t and not account for scr refresh
            calibration_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_start_text.started')
            # update status
            calibration_start_text.status = STARTED
            calibration_start_text.setAutoDraw(True)
        
        # if calibration_start_text is active this frame...
        if calibration_start_text.status == STARTED:
            # update params
            pass
        # *calibration_start_mouse* updates
        
        # if calibration_start_mouse is starting this frame...
        if calibration_start_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_start_mouse.frameNStart = frameN  # exact frame index
            calibration_start_mouse.tStart = t  # local t and not account for scr refresh
            calibration_start_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_start_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('calibration_start_mouse.started', t)
            # update status
            calibration_start_mouse.status = STARTED
            calibration_start_mouse.mouseClock.reset()
            prevButtonState = calibration_start_mouse.getPressed()  # if button is down already this ISN'T a new click
        if calibration_start_mouse.status == STARTED:  # only update if started and not finished!
            buttons = calibration_start_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = calibration_start_mouse.getPos()
                    calibration_start_mouse.x.append(x)
                    calibration_start_mouse.y.append(y)
                    buttons = calibration_start_mouse.getPressed()
                    calibration_start_mouse.leftButton.append(buttons[0])
                    calibration_start_mouse.midButton.append(buttons[1])
                    calibration_start_mouse.rightButton.append(buttons[2])
                    calibration_start_mouse.time.append(calibration_start_mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *calibration_start_key* updates
        
        # if calibration_start_key is starting this frame...
        if calibration_start_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_start_key.frameNStart = frameN  # exact frame index
            calibration_start_key.tStart = t  # local t and not account for scr refresh
            calibration_start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('calibration_start_key.started', t)
            # update status
            calibration_start_key.status = STARTED
            # keyboard checking is just starting
            calibration_start_key.clock.reset()  # now t=0
            calibration_start_key.clearEvents(eventType='keyboard')
        if calibration_start_key.status == STARTED:
            theseKeys = calibration_start_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _calibration_start_key_allKeys.extend(theseKeys)
            if len(_calibration_start_key_allKeys):
                calibration_start_key.keys = _calibration_start_key_allKeys[-1].name  # just the last key pressed
                calibration_start_key.rt = _calibration_start_key_allKeys[-1].rt
                calibration_start_key.duration = _calibration_start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_start" ---
    for thisComponent in calibration_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_start
    calibration_start.tStop = globalClock.getTime(format='float')
    calibration_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_start.stopped', calibration_start.tStop)
    # Run 'End Routine' code from calibration_start_code
    camera_view.show()
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('calibration_start_mouse.x', calibration_start_mouse.x)
    thisExp.addData('calibration_start_mouse.y', calibration_start_mouse.y)
    thisExp.addData('calibration_start_mouse.leftButton', calibration_start_mouse.leftButton)
    thisExp.addData('calibration_start_mouse.midButton', calibration_start_mouse.midButton)
    thisExp.addData('calibration_start_mouse.rightButton', calibration_start_mouse.rightButton)
    thisExp.addData('calibration_start_mouse.time', calibration_start_mouse.time)
    # check responses
    if calibration_start_key.keys in ['', [], None]:  # No response was made
        calibration_start_key.keys = None
    thisExp.addData('calibration_start_key.keys',calibration_start_key.keys)
    if calibration_start_key.keys != None:  # we had a response
        thisExp.addData('calibration_start_key.rt', calibration_start_key.rt)
        thisExp.addData('calibration_start_key.duration', calibration_start_key.duration)
    thisExp.nextEntry()
    # the Routine "calibration_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_appearance_windows" ---
    # create an object to store info about Routine calibration_appearance_windows
    calibration_appearance_windows = data.Routine(
        name='calibration_appearance_windows',
        components=[calibration_appearance_windows_text, calibration_appearance_windows_prompt],
    )
    calibration_appearance_windows.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from calibration_appearance_windows_code
    stylizer.reset(style='windows')
    base_window.reset()
    base_window.allocate_target(index=0)
    base_window.update_title("Appearance (Windows Style)")
    task_bar.reset()
    task_bar.show()
    base_window.show()
    calibration_appearance_windows_prompt.reset()
    calibration_appearance_windows_prompt.setPos((0, 1.0 - FONT_SIZE / 2))
    calibration_appearance_windows_prompt.setSize((2.0, FONT_SIZE))
    # store start times for calibration_appearance_windows
    calibration_appearance_windows.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_appearance_windows.tStart = globalClock.getTime(format='float')
    calibration_appearance_windows.status = STARTED
    thisExp.addData('calibration_appearance_windows.started', calibration_appearance_windows.tStart)
    calibration_appearance_windows.maxDuration = None
    # keep track of which components have finished
    calibration_appearance_windowsComponents = calibration_appearance_windows.components
    for thisComponent in calibration_appearance_windows.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_appearance_windows" ---
    calibration_appearance_windows.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from calibration_appearance_windows_code
        task_bar.update()
        continueRoutine = base_window.update()
        
        # *calibration_appearance_windows_text* updates
        
        # if calibration_appearance_windows_text is starting this frame...
        if calibration_appearance_windows_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_appearance_windows_text.frameNStart = frameN  # exact frame index
            calibration_appearance_windows_text.tStart = t  # local t and not account for scr refresh
            calibration_appearance_windows_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_appearance_windows_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_appearance_windows_text.started')
            # update status
            calibration_appearance_windows_text.status = STARTED
            calibration_appearance_windows_text.setAutoDraw(True)
        
        # if calibration_appearance_windows_text is active this frame...
        if calibration_appearance_windows_text.status == STARTED:
            # update params
            pass
        
        # *calibration_appearance_windows_prompt* updates
        
        # if calibration_appearance_windows_prompt is starting this frame...
        if calibration_appearance_windows_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_appearance_windows_prompt.frameNStart = frameN  # exact frame index
            calibration_appearance_windows_prompt.tStart = t  # local t and not account for scr refresh
            calibration_appearance_windows_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_appearance_windows_prompt, 'tStartRefresh')  # time at next scr refresh
            # update status
            calibration_appearance_windows_prompt.status = STARTED
            calibration_appearance_windows_prompt.setAutoDraw(True)
        
        # if calibration_appearance_windows_prompt is active this frame...
        if calibration_appearance_windows_prompt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_appearance_windows.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_appearance_windows.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_appearance_windows" ---
    for thisComponent in calibration_appearance_windows.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_appearance_windows
    calibration_appearance_windows.tStop = globalClock.getTime(format='float')
    calibration_appearance_windows.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_appearance_windows.stopped', calibration_appearance_windows.tStop)
    # Run 'End Routine' code from calibration_appearance_windows_code
    base_window.log()
    stylizer.log()
    hide_all()
    thisExp.nextEntry()
    # the Routine "calibration_appearance_windows" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_appearance_mac" ---
    # create an object to store info about Routine calibration_appearance_mac
    calibration_appearance_mac = data.Routine(
        name='calibration_appearance_mac',
        components=[calibration_appearance_mac_text, calibration_appearance_mac_prompt],
    )
    calibration_appearance_mac.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from calibration_appearance_mac_code
    stylizer.reset(style='mac')
    base_window.reset()
    base_window.allocate_target(index=0)
    base_window.update_title("Appearance (macOS Style)")
    task_bar.reset()
    task_bar.show()
    base_window.show()
    calibration_appearance_mac_prompt.reset()
    calibration_appearance_mac_prompt.setPos((0, 1.0 - FONT_SIZE / 2))
    calibration_appearance_mac_prompt.setSize((2.0, FONT_SIZE))
    # store start times for calibration_appearance_mac
    calibration_appearance_mac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_appearance_mac.tStart = globalClock.getTime(format='float')
    calibration_appearance_mac.status = STARTED
    thisExp.addData('calibration_appearance_mac.started', calibration_appearance_mac.tStart)
    calibration_appearance_mac.maxDuration = None
    # keep track of which components have finished
    calibration_appearance_macComponents = calibration_appearance_mac.components
    for thisComponent in calibration_appearance_mac.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_appearance_mac" ---
    calibration_appearance_mac.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from calibration_appearance_mac_code
        task_bar.update()
        continueRoutine = base_window.update()
        
        # *calibration_appearance_mac_text* updates
        
        # if calibration_appearance_mac_text is starting this frame...
        if calibration_appearance_mac_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_appearance_mac_text.frameNStart = frameN  # exact frame index
            calibration_appearance_mac_text.tStart = t  # local t and not account for scr refresh
            calibration_appearance_mac_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_appearance_mac_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_appearance_mac_text.started')
            # update status
            calibration_appearance_mac_text.status = STARTED
            calibration_appearance_mac_text.setAutoDraw(True)
        
        # if calibration_appearance_mac_text is active this frame...
        if calibration_appearance_mac_text.status == STARTED:
            # update params
            pass
        
        # *calibration_appearance_mac_prompt* updates
        
        # if calibration_appearance_mac_prompt is starting this frame...
        if calibration_appearance_mac_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_appearance_mac_prompt.frameNStart = frameN  # exact frame index
            calibration_appearance_mac_prompt.tStart = t  # local t and not account for scr refresh
            calibration_appearance_mac_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_appearance_mac_prompt, 'tStartRefresh')  # time at next scr refresh
            # update status
            calibration_appearance_mac_prompt.status = STARTED
            calibration_appearance_mac_prompt.setAutoDraw(True)
        
        # if calibration_appearance_mac_prompt is active this frame...
        if calibration_appearance_mac_prompt.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_appearance_mac.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_appearance_mac.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_appearance_mac" ---
    for thisComponent in calibration_appearance_mac.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_appearance_mac
    calibration_appearance_mac.tStop = globalClock.getTime(format='float')
    calibration_appearance_mac.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_appearance_mac.stopped', calibration_appearance_mac.tStop)
    # Run 'End Routine' code from calibration_appearance_mac_code
    base_window.log()
    stylizer.log()
    hide_all()
    thisExp.nextEntry()
    # the Routine "calibration_appearance_mac" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_typing_start" ---
    # create an object to store info about Routine calibration_typing_start
    calibration_typing_start = data.Routine(
        name='calibration_typing_start',
        components=[calibration_typing_start_text, calibration_typing_start_key],
    )
    calibration_typing_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for calibration_typing_start_key
    calibration_typing_start_key.keys = []
    calibration_typing_start_key.rt = []
    _calibration_typing_start_key_allKeys = []
    # store start times for calibration_typing_start
    calibration_typing_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_typing_start.tStart = globalClock.getTime(format='float')
    calibration_typing_start.status = STARTED
    thisExp.addData('calibration_typing_start.started', calibration_typing_start.tStart)
    calibration_typing_start.maxDuration = None
    # keep track of which components have finished
    calibration_typing_startComponents = calibration_typing_start.components
    for thisComponent in calibration_typing_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_typing_start" ---
    calibration_typing_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *calibration_typing_start_text* updates
        
        # if calibration_typing_start_text is starting this frame...
        if calibration_typing_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_typing_start_text.frameNStart = frameN  # exact frame index
            calibration_typing_start_text.tStart = t  # local t and not account for scr refresh
            calibration_typing_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_typing_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_typing_start_text.started')
            # update status
            calibration_typing_start_text.status = STARTED
            calibration_typing_start_text.setAutoDraw(True)
        
        # if calibration_typing_start_text is active this frame...
        if calibration_typing_start_text.status == STARTED:
            # update params
            pass
        
        # *calibration_typing_start_key* updates
        waitOnFlip = False
        
        # if calibration_typing_start_key is starting this frame...
        if calibration_typing_start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_typing_start_key.frameNStart = frameN  # exact frame index
            calibration_typing_start_key.tStart = t  # local t and not account for scr refresh
            calibration_typing_start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_typing_start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_typing_start_key.started')
            # update status
            calibration_typing_start_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(calibration_typing_start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(calibration_typing_start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if calibration_typing_start_key.status == STARTED and not waitOnFlip:
            theseKeys = calibration_typing_start_key.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _calibration_typing_start_key_allKeys.extend(theseKeys)
            if len(_calibration_typing_start_key_allKeys):
                calibration_typing_start_key.keys = _calibration_typing_start_key_allKeys[-1].name  # just the last key pressed
                calibration_typing_start_key.rt = _calibration_typing_start_key_allKeys[-1].rt
                calibration_typing_start_key.duration = _calibration_typing_start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_typing_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_typing_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_typing_start" ---
    for thisComponent in calibration_typing_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_typing_start
    calibration_typing_start.tStop = globalClock.getTime(format='float')
    calibration_typing_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_typing_start.stopped', calibration_typing_start.tStop)
    # check responses
    if calibration_typing_start_key.keys in ['', [], None]:  # No response was made
        calibration_typing_start_key.keys = None
    thisExp.addData('calibration_typing_start_key.keys',calibration_typing_start_key.keys)
    if calibration_typing_start_key.keys != None:  # we had a response
        thisExp.addData('calibration_typing_start_key.rt', calibration_typing_start_key.rt)
        thisExp.addData('calibration_typing_start_key.duration', calibration_typing_start_key.duration)
    thisExp.nextEntry()
    # the Routine "calibration_typing_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_typing" ---
    # create an object to store info about Routine calibration_typing
    calibration_typing = data.Routine(
        name='calibration_typing',
        components=[calibration_typing_key],
    )
    calibration_typing.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from calibration_typing_code
    stylizer.reset(style='windows')
    base_window.reset()
    base_window.update_title("Typing Speed Test")
    single_view_editor.allocate_test_target()
    task_bar.reset()
    task_bar.show()
    base_window.show()
    single_view_editor.show()
    
    win.winHandle.activate()
    
    calibration_typing_clock = core.MonotonicClock()
    # create starting attributes for calibration_typing_key
    calibration_typing_key.keys = []
    calibration_typing_key.rt = []
    _calibration_typing_key_allKeys = []
    # store start times for calibration_typing
    calibration_typing.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_typing.tStart = globalClock.getTime(format='float')
    calibration_typing.status = STARTED
    thisExp.addData('calibration_typing.started', calibration_typing.tStart)
    calibration_typing.maxDuration = None
    # keep track of which components have finished
    calibration_typingComponents = calibration_typing.components
    for thisComponent in calibration_typing.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_typing" ---
    calibration_typing.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from calibration_typing_code
        base_window.update()
        task_bar.update()
        continueRoutine = single_view_editor.update()
        
        
        # *calibration_typing_key* updates
        
        # if calibration_typing_key is starting this frame...
        if calibration_typing_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_typing_key.frameNStart = frameN  # exact frame index
            calibration_typing_key.tStart = t  # local t and not account for scr refresh
            calibration_typing_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_typing_key, 'tStartRefresh')  # time at next scr refresh
            # update status
            calibration_typing_key.status = STARTED
            # keyboard checking is just starting
            calibration_typing_key.clock.reset()  # now t=0
        if calibration_typing_key.status == STARTED:
            theseKeys = calibration_typing_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=True)
            _calibration_typing_key_allKeys.extend(theseKeys)
            if len(_calibration_typing_key_allKeys):
                calibration_typing_key.keys = [key.name for key in _calibration_typing_key_allKeys]  # storing all keys
                calibration_typing_key.rt = [key.rt for key in _calibration_typing_key_allKeys]
                calibration_typing_key.duration = [key.duration for key in _calibration_typing_key_allKeys]
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_typing.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_typing.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_typing" ---
    for thisComponent in calibration_typing.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_typing
    calibration_typing.tStop = globalClock.getTime(format='float')
    calibration_typing.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_typing.stopped', calibration_typing.tStop)
    # Run 'End Routine' code from calibration_typing_code
    single_view_editor.log()
    personalized_settings["typing_speed"] = len(single_view_editor.TARGET) / calibration_typing_clock.getTime()
    hide_all()
    
    # check responses
    if calibration_typing_key.keys in ['', [], None]:  # No response was made
        calibration_typing_key.keys = None
    thisExp.addData('calibration_typing_key.keys',calibration_typing_key.keys)
    if calibration_typing_key.keys != None:  # we had a response
        thisExp.addData('calibration_typing_key.rt', calibration_typing_key.rt)
        thisExp.addData('calibration_typing_key.duration', calibration_typing_key.duration)
    thisExp.nextEntry()
    # the Routine "calibration_typing" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_end" ---
    # create an object to store info about Routine calibration_end
    calibration_end = data.Routine(
        name='calibration_end',
        components=[calibration_end_text, calibration_end_mouse, calibration_end_key, calibration_end_debug_text],
    )
    calibration_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from calibration_end_code
    print(personalized_settings)
    camera_view.hide()
    
    serial_connector.write(SerialConnector.CALIB_END)
    # setup some python lists for storing info about the calibration_end_mouse
    calibration_end_mouse.x = []
    calibration_end_mouse.y = []
    calibration_end_mouse.leftButton = []
    calibration_end_mouse.midButton = []
    calibration_end_mouse.rightButton = []
    calibration_end_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for calibration_end_key
    calibration_end_key.keys = []
    calibration_end_key.rt = []
    _calibration_end_key_allKeys = []
    calibration_end_debug_text.setText("Typing Speed (char/sec): " + "{:.2f}".format(personalized_settings["typing_speed"]))
    # store start times for calibration_end
    calibration_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_end.tStart = globalClock.getTime(format='float')
    calibration_end.status = STARTED
    thisExp.addData('calibration_end.started', calibration_end.tStart)
    calibration_end.maxDuration = None
    # keep track of which components have finished
    calibration_endComponents = calibration_end.components
    for thisComponent in calibration_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_end" ---
    calibration_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *calibration_end_text* updates
        
        # if calibration_end_text is starting this frame...
        if calibration_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_end_text.frameNStart = frameN  # exact frame index
            calibration_end_text.tStart = t  # local t and not account for scr refresh
            calibration_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_end_text.started')
            # update status
            calibration_end_text.status = STARTED
            calibration_end_text.setAutoDraw(True)
        
        # if calibration_end_text is active this frame...
        if calibration_end_text.status == STARTED:
            # update params
            pass
        # *calibration_end_mouse* updates
        
        # if calibration_end_mouse is starting this frame...
        if calibration_end_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_end_mouse.frameNStart = frameN  # exact frame index
            calibration_end_mouse.tStart = t  # local t and not account for scr refresh
            calibration_end_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_end_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('calibration_end_mouse.started', t)
            # update status
            calibration_end_mouse.status = STARTED
            calibration_end_mouse.mouseClock.reset()
            prevButtonState = calibration_end_mouse.getPressed()  # if button is down already this ISN'T a new click
        if calibration_end_mouse.status == STARTED:  # only update if started and not finished!
            buttons = calibration_end_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = calibration_end_mouse.getPos()
                    calibration_end_mouse.x.append(x)
                    calibration_end_mouse.y.append(y)
                    buttons = calibration_end_mouse.getPressed()
                    calibration_end_mouse.leftButton.append(buttons[0])
                    calibration_end_mouse.midButton.append(buttons[1])
                    calibration_end_mouse.rightButton.append(buttons[2])
                    calibration_end_mouse.time.append(calibration_end_mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *calibration_end_key* updates
        
        # if calibration_end_key is starting this frame...
        if calibration_end_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_end_key.frameNStart = frameN  # exact frame index
            calibration_end_key.tStart = t  # local t and not account for scr refresh
            calibration_end_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_end_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('calibration_end_key.started', t)
            # update status
            calibration_end_key.status = STARTED
            # keyboard checking is just starting
            calibration_end_key.clock.reset()  # now t=0
            calibration_end_key.clearEvents(eventType='keyboard')
        if calibration_end_key.status == STARTED:
            theseKeys = calibration_end_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _calibration_end_key_allKeys.extend(theseKeys)
            if len(_calibration_end_key_allKeys):
                calibration_end_key.keys = _calibration_end_key_allKeys[-1].name  # just the last key pressed
                calibration_end_key.rt = _calibration_end_key_allKeys[-1].rt
                calibration_end_key.duration = _calibration_end_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *calibration_end_debug_text* updates
        
        # if calibration_end_debug_text is starting this frame...
        if calibration_end_debug_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            calibration_end_debug_text.frameNStart = frameN  # exact frame index
            calibration_end_debug_text.tStart = t  # local t and not account for scr refresh
            calibration_end_debug_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(calibration_end_debug_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'calibration_end_debug_text.started')
            # update status
            calibration_end_debug_text.status = STARTED
            calibration_end_debug_text.setAutoDraw(True)
        
        # if calibration_end_debug_text is active this frame...
        if calibration_end_debug_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            calibration_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibration_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_end" ---
    for thisComponent in calibration_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_end
    calibration_end.tStop = globalClock.getTime(format='float')
    calibration_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_end.stopped', calibration_end.tStop)
    # Run 'End Routine' code from calibration_end_code
    for key_name, key_value in personalized_settings.items():
        thisExp.addData(key_name, key_value)
    
    camera_connector.save()
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('calibration_end_mouse.x', calibration_end_mouse.x)
    thisExp.addData('calibration_end_mouse.y', calibration_end_mouse.y)
    thisExp.addData('calibration_end_mouse.leftButton', calibration_end_mouse.leftButton)
    thisExp.addData('calibration_end_mouse.midButton', calibration_end_mouse.midButton)
    thisExp.addData('calibration_end_mouse.rightButton', calibration_end_mouse.rightButton)
    thisExp.addData('calibration_end_mouse.time', calibration_end_mouse.time)
    # check responses
    if calibration_end_key.keys in ['', [], None]:  # No response was made
        calibration_end_key.keys = None
    thisExp.addData('calibration_end_key.keys',calibration_end_key.keys)
    if calibration_end_key.keys != None:  # we had a response
        thisExp.addData('calibration_end_key.rt', calibration_end_key.rt)
        thisExp.addData('calibration_end_key.duration', calibration_end_key.duration)
    thisExp.nextEntry()
    # the Routine "calibration_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "experiment_start" ---
    # create an object to store info about Routine experiment_start
    experiment_start = data.Routine(
        name='experiment_start',
        components=[experiment_start_text, experiment_start_mouse, experiment_start_key],
    )
    experiment_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from experiment_start_code
    serial_connector.write(SerialConnector.EXP_BEGIN)
    # setup some python lists for storing info about the experiment_start_mouse
    experiment_start_mouse.x = []
    experiment_start_mouse.y = []
    experiment_start_mouse.leftButton = []
    experiment_start_mouse.midButton = []
    experiment_start_mouse.rightButton = []
    experiment_start_mouse.time = []
    gotValidClick = False  # until a click is received
    # create starting attributes for experiment_start_key
    experiment_start_key.keys = []
    experiment_start_key.rt = []
    _experiment_start_key_allKeys = []
    # store start times for experiment_start
    experiment_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    experiment_start.tStart = globalClock.getTime(format='float')
    experiment_start.status = STARTED
    thisExp.addData('experiment_start.started', experiment_start.tStart)
    experiment_start.maxDuration = None
    # keep track of which components have finished
    experiment_startComponents = experiment_start.components
    for thisComponent in experiment_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "experiment_start" ---
    experiment_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *experiment_start_text* updates
        
        # if experiment_start_text is starting this frame...
        if experiment_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_start_text.frameNStart = frameN  # exact frame index
            experiment_start_text.tStart = t  # local t and not account for scr refresh
            experiment_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'experiment_start_text.started')
            # update status
            experiment_start_text.status = STARTED
            experiment_start_text.setAutoDraw(True)
        
        # if experiment_start_text is active this frame...
        if experiment_start_text.status == STARTED:
            # update params
            pass
        # *experiment_start_mouse* updates
        
        # if experiment_start_mouse is starting this frame...
        if experiment_start_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_start_mouse.frameNStart = frameN  # exact frame index
            experiment_start_mouse.tStart = t  # local t and not account for scr refresh
            experiment_start_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_start_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('experiment_start_mouse.started', t)
            # update status
            experiment_start_mouse.status = STARTED
            experiment_start_mouse.mouseClock.reset()
            prevButtonState = experiment_start_mouse.getPressed()  # if button is down already this ISN'T a new click
        if experiment_start_mouse.status == STARTED:  # only update if started and not finished!
            buttons = experiment_start_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = experiment_start_mouse.getPos()
                    experiment_start_mouse.x.append(x)
                    experiment_start_mouse.y.append(y)
                    buttons = experiment_start_mouse.getPressed()
                    experiment_start_mouse.leftButton.append(buttons[0])
                    experiment_start_mouse.midButton.append(buttons[1])
                    experiment_start_mouse.rightButton.append(buttons[2])
                    experiment_start_mouse.time.append(experiment_start_mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *experiment_start_key* updates
        
        # if experiment_start_key is starting this frame...
        if experiment_start_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_start_key.frameNStart = frameN  # exact frame index
            experiment_start_key.tStart = t  # local t and not account for scr refresh
            experiment_start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('experiment_start_key.started', t)
            # update status
            experiment_start_key.status = STARTED
            # keyboard checking is just starting
            experiment_start_key.clock.reset()  # now t=0
            experiment_start_key.clearEvents(eventType='keyboard')
        if experiment_start_key.status == STARTED:
            theseKeys = experiment_start_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _experiment_start_key_allKeys.extend(theseKeys)
            if len(_experiment_start_key_allKeys):
                experiment_start_key.keys = _experiment_start_key_allKeys[-1].name  # just the last key pressed
                experiment_start_key.rt = _experiment_start_key_allKeys[-1].rt
                experiment_start_key.duration = _experiment_start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            experiment_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in experiment_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "experiment_start" ---
    for thisComponent in experiment_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for experiment_start
    experiment_start.tStop = globalClock.getTime(format='float')
    experiment_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('experiment_start.stopped', experiment_start.tStop)
    # Run 'End Routine' code from experiment_start_code
    event_timer = EventTimer()
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('experiment_start_mouse.x', experiment_start_mouse.x)
    thisExp.addData('experiment_start_mouse.y', experiment_start_mouse.y)
    thisExp.addData('experiment_start_mouse.leftButton', experiment_start_mouse.leftButton)
    thisExp.addData('experiment_start_mouse.midButton', experiment_start_mouse.midButton)
    thisExp.addData('experiment_start_mouse.rightButton', experiment_start_mouse.rightButton)
    thisExp.addData('experiment_start_mouse.time', experiment_start_mouse.time)
    # check responses
    if experiment_start_key.keys in ['', [], None]:  # No response was made
        experiment_start_key.keys = None
    thisExp.addData('experiment_start_key.keys',experiment_start_key.keys)
    if experiment_start_key.keys != None:  # we had a response
        thisExp.addData('experiment_start_key.rt', experiment_start_key.rt)
        thisExp.addData('experiment_start_key.duration', experiment_start_key.duration)
    thisExp.nextEntry()
    # the Routine "experiment_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=60.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "style_randomizer" ---
        # create an object to store info about Routine style_randomizer
        style_randomizer = data.Routine(
            name='style_randomizer',
            components=[],
        )
        style_randomizer.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from style_randomizer_code
        reset_layout()
        camera_connector.record()
        # store start times for style_randomizer
        style_randomizer.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        style_randomizer.tStart = globalClock.getTime(format='float')
        style_randomizer.status = STARTED
        thisExp.addData('style_randomizer.started', style_randomizer.tStart)
        style_randomizer.maxDuration = None
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        style_randomizerComponents = style_randomizer.components
        for thisComponent in style_randomizer.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "style_randomizer" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        style_randomizer.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                style_randomizer.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in style_randomizer.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "style_randomizer" ---
        for thisComponent in style_randomizer.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for style_randomizer
        style_randomizer.tStop = globalClock.getTime(format='float')
        style_randomizer.tStopRefresh = tThisFlipGlobal
        thisExp.addData('style_randomizer.stopped', style_randomizer.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from style_randomizer_code
        stylizer.log()
        camera_view.show()
        if event_timer.should_end():
            trials.finished = True
        # the Routine "style_randomizer" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        tasks = data.TrialHandler2(
            name='tasks',
            nReps=5.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(tasks)  # add the loop to the experiment
        thisTask = tasks.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTask.rgb)
        if thisTask != None:
            for paramName in thisTask:
                globals()[paramName] = thisTask[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTask in tasks:
            currentLoop = tasks
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTask.rgb)
            if thisTask != None:
                for paramName in thisTask:
                    globals()[paramName] = thisTask[paramName]
            
            # set up handler to look after randomisation of conditions etc
            mail = data.TrialHandler2(
                name='mail',
                nReps=event_timer.should_show('MAIL'), 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(mail)  # add the loop to the experiment
            thisMail = mail.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisMail.rgb)
            if thisMail != None:
                for paramName in thisMail:
                    globals()[paramName] = thisMail[paramName]
            
            for thisMail in mail:
                currentLoop = mail
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisMail.rgb)
                if thisMail != None:
                    for paramName in thisMail:
                        globals()[paramName] = thisMail[paramName]
                
                # --- Prepare to start Routine "mail_homescreen" ---
                # create an object to store info about Routine mail_homescreen
                mail_homescreen = data.Routine(
                    name='mail_homescreen',
                    components=[mail_mouse_homescreen, mail_textbox_homescreen],
                )
                mail_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from mail_code_homescreen
                task_bar.allocate_target(index=2)
                task_bar.show()
                serial_connector.write(SerialConnector.MAIL_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the mail_mouse_homescreen
                mail_mouse_homescreen.x = []
                mail_mouse_homescreen.y = []
                mail_mouse_homescreen.leftButton = []
                mail_mouse_homescreen.midButton = []
                mail_mouse_homescreen.rightButton = []
                mail_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                mail_mouse_homescreen.mouseClock.reset()
                mail_textbox_homescreen.reset()
                mail_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                mail_textbox_homescreen.setSize((2.0, FONT_SIZE))
                mail_textbox_homescreen.setText("Wait for <i>{} Notification</i>".format(task_bar.target_name))
                # store start times for mail_homescreen
                mail_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                mail_homescreen.tStart = globalClock.getTime(format='float')
                mail_homescreen.status = STARTED
                thisExp.addData('mail_homescreen.started', mail_homescreen.tStart)
                mail_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                mail_homescreenComponents = mail_homescreen.components
                for thisComponent in mail_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "mail_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(mail, data.TrialHandler2) and thisMail.thisN != mail.thisTrial.thisN:
                    continueRoutine = False
                mail_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 5.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from mail_code_homescreen
                    continueRoutine = task_bar.update()
                    # *mail_mouse_homescreen* updates
                    
                    # if mail_mouse_homescreen is starting this frame...
                    if mail_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_mouse_homescreen.frameNStart = frameN  # exact frame index
                        mail_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        mail_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'mail_mouse_homescreen.started')
                        # update status
                        mail_mouse_homescreen.status = STARTED
                        prevButtonState = mail_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    
                    # if mail_mouse_homescreen is stopping this frame...
                    if mail_mouse_homescreen.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mail_mouse_homescreen.tStartRefresh + 5.0-frameTolerance:
                            # keep track of stop time/frame for later
                            mail_mouse_homescreen.tStop = t  # not accounting for scr refresh
                            mail_mouse_homescreen.tStopRefresh = tThisFlipGlobal  # on global time
                            mail_mouse_homescreen.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'mail_mouse_homescreen.stopped')
                            # update status
                            mail_mouse_homescreen.status = FINISHED
                    if mail_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = mail_mouse_homescreen.getPos()
                        mail_mouse_homescreen.x.append(x)
                        mail_mouse_homescreen.y.append(y)
                        buttons = mail_mouse_homescreen.getPressed()
                        mail_mouse_homescreen.leftButton.append(buttons[0])
                        mail_mouse_homescreen.midButton.append(buttons[1])
                        mail_mouse_homescreen.rightButton.append(buttons[2])
                        mail_mouse_homescreen.time.append(mail_mouse_homescreen.mouseClock.getTime())
                    
                    # *mail_textbox_homescreen* updates
                    
                    # if mail_textbox_homescreen is starting this frame...
                    if mail_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_textbox_homescreen.frameNStart = frameN  # exact frame index
                        mail_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        mail_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mail_textbox_homescreen.status = STARTED
                        mail_textbox_homescreen.setAutoDraw(True)
                    
                    # if mail_textbox_homescreen is active this frame...
                    if mail_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # if mail_textbox_homescreen is stopping this frame...
                    if mail_textbox_homescreen.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > mail_textbox_homescreen.tStartRefresh + 5.0-frameTolerance:
                            # keep track of stop time/frame for later
                            mail_textbox_homescreen.tStop = t  # not accounting for scr refresh
                            mail_textbox_homescreen.tStopRefresh = tThisFlipGlobal  # on global time
                            mail_textbox_homescreen.frameNStop = frameN  # exact frame index
                            # update status
                            mail_textbox_homescreen.status = FINISHED
                            mail_textbox_homescreen.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        mail_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in mail_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "mail_homescreen" ---
                for thisComponent in mail_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for mail_homescreen
                mail_homescreen.tStop = globalClock.getTime(format='float')
                mail_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('mail_homescreen.stopped', mail_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from mail_code_homescreen
                #serial_connector.write(SerialConnector.MAIL_HOMESCREEN_END)
                # store data for mail (TrialHandler)
                mail.addData('mail_mouse_homescreen.x', mail_mouse_homescreen.x)
                mail.addData('mail_mouse_homescreen.y', mail_mouse_homescreen.y)
                mail.addData('mail_mouse_homescreen.leftButton', mail_mouse_homescreen.leftButton)
                mail.addData('mail_mouse_homescreen.midButton', mail_mouse_homescreen.midButton)
                mail.addData('mail_mouse_homescreen.rightButton', mail_mouse_homescreen.rightButton)
                mail.addData('mail_mouse_homescreen.time', mail_mouse_homescreen.time)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if mail_homescreen.maxDurationReached:
                    routineTimer.addTime(-mail_homescreen.maxDuration)
                elif mail_homescreen.forceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-5.000000)
                
                # --- Prepare to start Routine "mail_notification" ---
                # create an object to store info about Routine mail_notification
                mail_notification = data.Routine(
                    name='mail_notification',
                    components=[mail_notification_mouse, mail_notification_textbox],
                )
                mail_notification.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from mail_notification_code
                notification.show()
                serial_connector.write(SerialConnector.MAIL_NOTIFICATION_BEGIN)
                # setup some python lists for storing info about the mail_notification_mouse
                mail_notification_mouse.x = []
                mail_notification_mouse.y = []
                mail_notification_mouse.leftButton = []
                mail_notification_mouse.midButton = []
                mail_notification_mouse.rightButton = []
                mail_notification_mouse.time = []
                gotValidClick = False  # until a click is received
                mail_notification_mouse.mouseClock.reset()
                mail_notification_textbox.reset()
                mail_notification_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                mail_notification_textbox.setSize((2.0, FONT_SIZE))
                mail_notification_textbox.setText("Click on the <i>{} Notification Popup</i>".format(task_bar.target_name))
                # store start times for mail_notification
                mail_notification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                mail_notification.tStart = globalClock.getTime(format='float')
                mail_notification.status = STARTED
                thisExp.addData('mail_notification.started', mail_notification.tStart)
                mail_notification.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                mail_notificationComponents = mail_notification.components
                for thisComponent in mail_notification.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "mail_notification" ---
                # if trial has changed, end Routine now
                if isinstance(mail, data.TrialHandler2) and thisMail.thisN != mail.thisTrial.thisN:
                    continueRoutine = False
                mail_notification.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from mail_notification_code
                    task_bar.update()
                    continueRoutine = notification.update()
                    # *mail_notification_mouse* updates
                    
                    # if mail_notification_mouse is starting this frame...
                    if mail_notification_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_notification_mouse.frameNStart = frameN  # exact frame index
                        mail_notification_mouse.tStart = t  # local t and not account for scr refresh
                        mail_notification_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_notification_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'mail_notification_mouse.started')
                        # update status
                        mail_notification_mouse.status = STARTED
                        prevButtonState = mail_notification_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if mail_notification_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = mail_notification_mouse.getPos()
                        mail_notification_mouse.x.append(x)
                        mail_notification_mouse.y.append(y)
                        buttons = mail_notification_mouse.getPressed()
                        mail_notification_mouse.leftButton.append(buttons[0])
                        mail_notification_mouse.midButton.append(buttons[1])
                        mail_notification_mouse.rightButton.append(buttons[2])
                        mail_notification_mouse.time.append(mail_notification_mouse.mouseClock.getTime())
                    
                    # *mail_notification_textbox* updates
                    
                    # if mail_notification_textbox is starting this frame...
                    if mail_notification_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_notification_textbox.frameNStart = frameN  # exact frame index
                        mail_notification_textbox.tStart = t  # local t and not account for scr refresh
                        mail_notification_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_notification_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mail_notification_textbox.status = STARTED
                        mail_notification_textbox.setAutoDraw(True)
                    
                    # if mail_notification_textbox is active this frame...
                    if mail_notification_textbox.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        mail_notification.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in mail_notification.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "mail_notification" ---
                for thisComponent in mail_notification.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for mail_notification
                mail_notification.tStop = globalClock.getTime(format='float')
                mail_notification.tStopRefresh = tThisFlipGlobal
                thisExp.addData('mail_notification.stopped', mail_notification.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from mail_notification_code
                notification.hide()
                #serial_connector.write(SerialConnector.MAIL_NOTIFICATION_END)
                # store data for mail (TrialHandler)
                mail.addData('mail_notification_mouse.x', mail_notification_mouse.x)
                mail.addData('mail_notification_mouse.y', mail_notification_mouse.y)
                mail.addData('mail_notification_mouse.leftButton', mail_notification_mouse.leftButton)
                mail.addData('mail_notification_mouse.midButton', mail_notification_mouse.midButton)
                mail.addData('mail_notification_mouse.rightButton', mail_notification_mouse.rightButton)
                mail.addData('mail_notification_mouse.time', mail_notification_mouse.time)
                # the Routine "mail_notification" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "mail_content" ---
                # create an object to store info about Routine mail_content
                mail_content = data.Routine(
                    name='mail_content',
                    components=[mail_content_textbox, mail_content_user_key_release],
                )
                mail_content.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from mail_content_code
                base_window.update_title(task_bar.target_name)
                single_view_editor.allocate_target()
                base_window.show()
                single_view_editor.show()
                task_bar.show()
                
                win.winHandle.activate()
                
                serial_connector.write(SerialConnector.MAIL_CONTENT_BEGIN)
                mail_content_textbox.reset()
                mail_content_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                mail_content_textbox.setSize((2.0, FONT_SIZE))
                mail_content_textbox.setText("Please complete the <i>{}</i>".format(task_bar.target_name))
                # create starting attributes for mail_content_user_key_release
                mail_content_user_key_release.keys = []
                mail_content_user_key_release.rt = []
                _mail_content_user_key_release_allKeys = []
                # store start times for mail_content
                mail_content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                mail_content.tStart = globalClock.getTime(format='float')
                mail_content.status = STARTED
                thisExp.addData('mail_content.started', mail_content.tStart)
                mail_content.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                mail_contentComponents = mail_content.components
                for thisComponent in mail_content.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "mail_content" ---
                # if trial has changed, end Routine now
                if isinstance(mail, data.TrialHandler2) and thisMail.thisN != mail.thisTrial.thisN:
                    continueRoutine = False
                mail_content.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from mail_content_code
                    base_window.update()
                    task_bar.update()
                    continueRoutine = single_view_editor.update()
                    
                    
                    # *mail_content_textbox* updates
                    
                    # if mail_content_textbox is starting this frame...
                    if mail_content_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_content_textbox.frameNStart = frameN  # exact frame index
                        mail_content_textbox.tStart = t  # local t and not account for scr refresh
                        mail_content_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_content_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mail_content_textbox.status = STARTED
                        mail_content_textbox.setAutoDraw(True)
                    
                    # if mail_content_textbox is active this frame...
                    if mail_content_textbox.status == STARTED:
                        # update params
                        pass
                    
                    # *mail_content_user_key_release* updates
                    
                    # if mail_content_user_key_release is starting this frame...
                    if mail_content_user_key_release.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mail_content_user_key_release.frameNStart = frameN  # exact frame index
                        mail_content_user_key_release.tStart = t  # local t and not account for scr refresh
                        mail_content_user_key_release.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mail_content_user_key_release, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mail_content_user_key_release.status = STARTED
                        # keyboard checking is just starting
                        mail_content_user_key_release.clock.reset()  # now t=0
                    if mail_content_user_key_release.status == STARTED:
                        theseKeys = mail_content_user_key_release.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=True)
                        _mail_content_user_key_release_allKeys.extend(theseKeys)
                        if len(_mail_content_user_key_release_allKeys):
                            mail_content_user_key_release.keys = [key.name for key in _mail_content_user_key_release_allKeys]  # storing all keys
                            mail_content_user_key_release.rt = [key.rt for key in _mail_content_user_key_release_allKeys]
                            mail_content_user_key_release.duration = [key.duration for key in _mail_content_user_key_release_allKeys]
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        mail_content.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in mail_content.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "mail_content" ---
                for thisComponent in mail_content.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for mail_content
                mail_content.tStop = globalClock.getTime(format='float')
                mail_content.tStopRefresh = tThisFlipGlobal
                thisExp.addData('mail_content.stopped', mail_content.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from mail_content_code
                single_view_editor.log()
                single_view_editor.hide()
                #serial_connector.write(SerialConnector.MAIL_CONTENT_END)
                # check responses
                if mail_content_user_key_release.keys in ['', [], None]:  # No response was made
                    mail_content_user_key_release.keys = None
                mail.addData('mail_content_user_key_release.keys',mail_content_user_key_release.keys)
                if mail_content_user_key_release.keys != None:  # we had a response
                    mail.addData('mail_content_user_key_release.rt', mail_content_user_key_release.rt)
                    mail.addData('mail_content_user_key_release.duration', mail_content_user_key_release.duration)
                # the Routine "mail_content" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(mail, data.TrialHandler2) and thisMail.thisN != mail.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for mail (TrialHandler)
                mail.addData('window_close_mouse.x', window_close_mouse.x)
                mail.addData('window_close_mouse.y', window_close_mouse.y)
                mail.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                mail.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                mail.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                mail.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('MAIL') repeats of 'mail'
            
            
            # set up handler to look after randomisation of conditions etc
            file_manager_dragging_task = data.TrialHandler2(
                name='file_manager_dragging_task',
                nReps=event_timer.should_show('FILE_DRAGGING'), 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(file_manager_dragging_task)  # add the loop to the experiment
            thisFile_manager_dragging_task = file_manager_dragging_task.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisFile_manager_dragging_task.rgb)
            if thisFile_manager_dragging_task != None:
                for paramName in thisFile_manager_dragging_task:
                    globals()[paramName] = thisFile_manager_dragging_task[paramName]
            
            for thisFile_manager_dragging_task in file_manager_dragging_task:
                currentLoop = file_manager_dragging_task
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisFile_manager_dragging_task.rgb)
                if thisFile_manager_dragging_task != None:
                    for paramName in thisFile_manager_dragging_task:
                        globals()[paramName] = thisFile_manager_dragging_task[paramName]
                
                # --- Prepare to start Routine "file_manager_homescreen" ---
                # create an object to store info about Routine file_manager_homescreen
                file_manager_homescreen = data.Routine(
                    name='file_manager_homescreen',
                    components=[file_manager_mouse_homescreen, file_manager_textbox_homescreen],
                )
                file_manager_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from file_manager_code_homescreen
                task_bar.allocate_target(index=0)
                task_bar.show()
                
                serial_connector.write(SerialConnector.FILE_MANAGER_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the file_manager_mouse_homescreen
                file_manager_mouse_homescreen.x = []
                file_manager_mouse_homescreen.y = []
                file_manager_mouse_homescreen.leftButton = []
                file_manager_mouse_homescreen.midButton = []
                file_manager_mouse_homescreen.rightButton = []
                file_manager_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                file_manager_mouse_homescreen.mouseClock.reset()
                file_manager_textbox_homescreen.reset()
                file_manager_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                file_manager_textbox_homescreen.setSize((2.0, FONT_SIZE))
                file_manager_textbox_homescreen.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
                # store start times for file_manager_homescreen
                file_manager_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                file_manager_homescreen.tStart = globalClock.getTime(format='float')
                file_manager_homescreen.status = STARTED
                thisExp.addData('file_manager_homescreen.started', file_manager_homescreen.tStart)
                file_manager_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                file_manager_homescreenComponents = file_manager_homescreen.components
                for thisComponent in file_manager_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "file_manager_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(file_manager_dragging_task, data.TrialHandler2) and thisFile_manager_dragging_task.thisN != file_manager_dragging_task.thisTrial.thisN:
                    continueRoutine = False
                file_manager_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from file_manager_code_homescreen
                    continueRoutine = task_bar.update()
                    # *file_manager_mouse_homescreen* updates
                    
                    # if file_manager_mouse_homescreen is starting this frame...
                    if file_manager_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        file_manager_mouse_homescreen.frameNStart = frameN  # exact frame index
                        file_manager_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        file_manager_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(file_manager_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'file_manager_mouse_homescreen.started')
                        # update status
                        file_manager_mouse_homescreen.status = STARTED
                        prevButtonState = file_manager_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    if file_manager_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = file_manager_mouse_homescreen.getPos()
                        file_manager_mouse_homescreen.x.append(x)
                        file_manager_mouse_homescreen.y.append(y)
                        buttons = file_manager_mouse_homescreen.getPressed()
                        file_manager_mouse_homescreen.leftButton.append(buttons[0])
                        file_manager_mouse_homescreen.midButton.append(buttons[1])
                        file_manager_mouse_homescreen.rightButton.append(buttons[2])
                        file_manager_mouse_homescreen.time.append(file_manager_mouse_homescreen.mouseClock.getTime())
                    
                    # *file_manager_textbox_homescreen* updates
                    
                    # if file_manager_textbox_homescreen is starting this frame...
                    if file_manager_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        file_manager_textbox_homescreen.frameNStart = frameN  # exact frame index
                        file_manager_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        file_manager_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(file_manager_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        file_manager_textbox_homescreen.status = STARTED
                        file_manager_textbox_homescreen.setAutoDraw(True)
                    
                    # if file_manager_textbox_homescreen is active this frame...
                    if file_manager_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        file_manager_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in file_manager_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "file_manager_homescreen" ---
                for thisComponent in file_manager_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for file_manager_homescreen
                file_manager_homescreen.tStop = globalClock.getTime(format='float')
                file_manager_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('file_manager_homescreen.stopped', file_manager_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from file_manager_code_homescreen
                
                #serial_connector.write(SerialConnector.FILE_MANAGER_HOMESCREEN_END)
                # store data for file_manager_dragging_task (TrialHandler)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.x', file_manager_mouse_homescreen.x)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.y', file_manager_mouse_homescreen.y)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.leftButton', file_manager_mouse_homescreen.leftButton)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.midButton', file_manager_mouse_homescreen.midButton)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.rightButton', file_manager_mouse_homescreen.rightButton)
                file_manager_dragging_task.addData('file_manager_mouse_homescreen.time', file_manager_mouse_homescreen.time)
                # the Routine "file_manager_homescreen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # set up handler to look after randomisation of conditions etc
                file_dragging = data.TrialHandler2(
                    name='file_dragging',
                    nReps=20.0, 
                    method='sequential', 
                    extraInfo=expInfo, 
                    originPath=-1, 
                    trialList=[None], 
                    seed=None, 
                )
                thisExp.addLoop(file_dragging)  # add the loop to the experiment
                thisFile_dragging = file_dragging.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisFile_dragging.rgb)
                if thisFile_dragging != None:
                    for paramName in thisFile_dragging:
                        globals()[paramName] = thisFile_dragging[paramName]
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                
                for thisFile_dragging in file_dragging:
                    currentLoop = file_dragging
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                    # abbreviate parameter names if possible (e.g. rgb = thisFile_dragging.rgb)
                    if thisFile_dragging != None:
                        for paramName in thisFile_dragging:
                            globals()[paramName] = thisFile_dragging[paramName]
                    
                    # --- Prepare to start Routine "file_manager_dragging" ---
                    # create an object to store info about Routine file_manager_dragging
                    file_manager_dragging = data.Routine(
                        name='file_manager_dragging',
                        components=[file_manager_dragging_textbox, stimuli_dragging_mouse, stimuli_dragging_target_image, stimuli_dragging_stimuli_image],
                    )
                    file_manager_dragging.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from file_manager_dragging_code
                    base_window.update_title(task_bar.target_name)
                    base_window.show()
                    task_bar.show()
                    
                    # Define size and location of the stimuli
                    size = 0.1
                    radius = 0.2
                    distance = 0.4
                    angle = np.random.uniform(0, 2 * np.pi)
                    loc_x, loc_y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)
                    target_x, target_y = loc_x + distance * np.cos(angle), loc_y + distance * np.sin(angle)
                    # Log clicked elements
                    last_clicked_offset = None
                    
                    serial_connector.write(SerialConnector.FILE_MANAGER_DRAGGING_BEGIN)
                    file_manager_dragging_textbox.reset()
                    file_manager_dragging_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                    file_manager_dragging_textbox.setSize((2.0, FONT_SIZE))
                    file_manager_dragging_textbox.setText("Please drag the folder to the destination".format())
                    # setup some python lists for storing info about the stimuli_dragging_mouse
                    stimuli_dragging_mouse.x = []
                    stimuli_dragging_mouse.y = []
                    stimuli_dragging_mouse.leftButton = []
                    stimuli_dragging_mouse.midButton = []
                    stimuli_dragging_mouse.rightButton = []
                    stimuli_dragging_mouse.time = []
                    gotValidClick = False  # until a click is received
                    stimuli_dragging_mouse.mouseClock.reset()
                    stimuli_dragging_target_image.setPos((target_x, target_y))
                    stimuli_dragging_target_image.setSize((size, size * BaseWindow.ASPECT_RATIO))
                    stimuli_dragging_stimuli_image.setPos((loc_x, loc_y))
                    stimuli_dragging_stimuli_image.setSize((size, size * BaseWindow.ASPECT_RATIO))
                    # store start times for file_manager_dragging
                    file_manager_dragging.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    file_manager_dragging.tStart = globalClock.getTime(format='float')
                    file_manager_dragging.status = STARTED
                    thisExp.addData('file_manager_dragging.started', file_manager_dragging.tStart)
                    file_manager_dragging.maxDuration = None
                    win.color = 'white'
                    win.colorSpace = 'rgb'
                    win.backgroundImage = ''
                    win.backgroundFit = 'none'
                    # keep track of which components have finished
                    file_manager_draggingComponents = file_manager_dragging.components
                    for thisComponent in file_manager_dragging.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "file_manager_dragging" ---
                    # if trial has changed, end Routine now
                    if isinstance(file_dragging, data.TrialHandler2) and thisFile_dragging.thisN != file_dragging.thisTrial.thisN:
                        continueRoutine = False
                    file_manager_dragging.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from file_manager_dragging_code
                        task_bar.update()
                        base_window.update()
                        
                        buttons = stimuli_dragging_mouse.getPressed()
                        if last_clicked_offset is None and stimuli_dragging_mouse.isPressedIn(stimuli_dragging_stimuli_image):
                            last_clicked_offset = stimuli_dragging_stimuli_image.pos - stimuli_dragging_mouse.getPos()
                        elif last_clicked_offset is not None and np.sum(buttons) > 0:
                            stimuli_dragging_stimuli_image.setPos(last_clicked_offset + stimuli_dragging_mouse.getPos())
                        if np.sum(buttons) == 0 and last_clicked_offset is not None:
                            if np.linalg.norm(stimuli_dragging_stimuli_image.pos - stimuli_dragging_target_image.pos) <= size:
                                continueRoutine = False
                            else:
                                last_clicked_offset = None
                                stimuli_dragging_stimuli_image.setPos((loc_x, loc_y))
                        
                        # *file_manager_dragging_textbox* updates
                        
                        # if file_manager_dragging_textbox is starting this frame...
                        if file_manager_dragging_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            file_manager_dragging_textbox.frameNStart = frameN  # exact frame index
                            file_manager_dragging_textbox.tStart = t  # local t and not account for scr refresh
                            file_manager_dragging_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(file_manager_dragging_textbox, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            file_manager_dragging_textbox.status = STARTED
                            file_manager_dragging_textbox.setAutoDraw(True)
                        
                        # if file_manager_dragging_textbox is active this frame...
                        if file_manager_dragging_textbox.status == STARTED:
                            # update params
                            pass
                        # *stimuli_dragging_mouse* updates
                        
                        # if stimuli_dragging_mouse is starting this frame...
                        if stimuli_dragging_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_dragging_mouse.frameNStart = frameN  # exact frame index
                            stimuli_dragging_mouse.tStart = t  # local t and not account for scr refresh
                            stimuli_dragging_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_dragging_mouse, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_dragging_mouse.started')
                            # update status
                            stimuli_dragging_mouse.status = STARTED
                            prevButtonState = stimuli_dragging_mouse.getPressed()  # if button is down already this ISN'T a new click
                        if stimuli_dragging_mouse.status == STARTED:  # only update if started and not finished!
                            x, y = stimuli_dragging_mouse.getPos()
                            stimuli_dragging_mouse.x.append(x)
                            stimuli_dragging_mouse.y.append(y)
                            buttons = stimuli_dragging_mouse.getPressed()
                            stimuli_dragging_mouse.leftButton.append(buttons[0])
                            stimuli_dragging_mouse.midButton.append(buttons[1])
                            stimuli_dragging_mouse.rightButton.append(buttons[2])
                            stimuli_dragging_mouse.time.append(stimuli_dragging_mouse.mouseClock.getTime())
                        
                        # *stimuli_dragging_target_image* updates
                        
                        # if stimuli_dragging_target_image is starting this frame...
                        if stimuli_dragging_target_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_dragging_target_image.frameNStart = frameN  # exact frame index
                            stimuli_dragging_target_image.tStart = t  # local t and not account for scr refresh
                            stimuli_dragging_target_image.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_dragging_target_image, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_dragging_target_image.started')
                            # update status
                            stimuli_dragging_target_image.status = STARTED
                            stimuli_dragging_target_image.setAutoDraw(True)
                        
                        # if stimuli_dragging_target_image is active this frame...
                        if stimuli_dragging_target_image.status == STARTED:
                            # update params
                            pass
                        
                        # *stimuli_dragging_stimuli_image* updates
                        
                        # if stimuli_dragging_stimuli_image is starting this frame...
                        if stimuli_dragging_stimuli_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_dragging_stimuli_image.frameNStart = frameN  # exact frame index
                            stimuli_dragging_stimuli_image.tStart = t  # local t and not account for scr refresh
                            stimuli_dragging_stimuli_image.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_dragging_stimuli_image, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_dragging_stimuli_image.started')
                            # update status
                            stimuli_dragging_stimuli_image.status = STARTED
                            stimuli_dragging_stimuli_image.setAutoDraw(True)
                        
                        # if stimuli_dragging_stimuli_image is active this frame...
                        if stimuli_dragging_stimuli_image.status == STARTED:
                            # update params
                            pass
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            file_manager_dragging.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in file_manager_dragging.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "file_manager_dragging" ---
                    for thisComponent in file_manager_dragging.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for file_manager_dragging
                    file_manager_dragging.tStop = globalClock.getTime(format='float')
                    file_manager_dragging.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('file_manager_dragging.stopped', file_manager_dragging.tStop)
                    setupWindow(expInfo=expInfo, win=win)
                    # Run 'End Routine' code from file_manager_dragging_code
                    thisExp.addData('size', size)
                    thisExp.addData('radius', radius)
                    thisExp.addData('angle', angle)
                    thisExp.addData('distance', distance)
                    thisExp.addData('stimuli_location', (loc_x, loc_y))
                    thisExp.addData('target_location', (target_x, target_y))
                    thisExp.addData('last_clicked_offset', tuple(last_clicked_offset))
                    
                    #serial_connector.write(SerialConnector.FILE_MANAGER_DRAGGING_END)
                    # store data for file_dragging (TrialHandler)
                    file_dragging.addData('stimuli_dragging_mouse.x', stimuli_dragging_mouse.x)
                    file_dragging.addData('stimuli_dragging_mouse.y', stimuli_dragging_mouse.y)
                    file_dragging.addData('stimuli_dragging_mouse.leftButton', stimuli_dragging_mouse.leftButton)
                    file_dragging.addData('stimuli_dragging_mouse.midButton', stimuli_dragging_mouse.midButton)
                    file_dragging.addData('stimuli_dragging_mouse.rightButton', stimuli_dragging_mouse.rightButton)
                    file_dragging.addData('stimuli_dragging_mouse.time', stimuli_dragging_mouse.time)
                    # the Routine "file_manager_dragging" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    thisExp.nextEntry()
                    
                # completed 20.0 repeats of 'file_dragging'
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(file_manager_dragging_task, data.TrialHandler2) and thisFile_manager_dragging_task.thisN != file_manager_dragging_task.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for file_manager_dragging_task (TrialHandler)
                file_manager_dragging_task.addData('window_close_mouse.x', window_close_mouse.x)
                file_manager_dragging_task.addData('window_close_mouse.y', window_close_mouse.y)
                file_manager_dragging_task.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                file_manager_dragging_task.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                file_manager_dragging_task.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                file_manager_dragging_task.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('FILE_DRAGGING') repeats of 'file_manager_dragging_task'
            
            
            # set up handler to look after randomisation of conditions etc
            file_manager_opening_task = data.TrialHandler2(
                name='file_manager_opening_task',
                nReps=event_timer.should_show('FILE_OPENING'), 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(file_manager_opening_task)  # add the loop to the experiment
            thisFile_manager_opening_task = file_manager_opening_task.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisFile_manager_opening_task.rgb)
            if thisFile_manager_opening_task != None:
                for paramName in thisFile_manager_opening_task:
                    globals()[paramName] = thisFile_manager_opening_task[paramName]
            
            for thisFile_manager_opening_task in file_manager_opening_task:
                currentLoop = file_manager_opening_task
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisFile_manager_opening_task.rgb)
                if thisFile_manager_opening_task != None:
                    for paramName in thisFile_manager_opening_task:
                        globals()[paramName] = thisFile_manager_opening_task[paramName]
                
                # --- Prepare to start Routine "file_manager_homescreen" ---
                # create an object to store info about Routine file_manager_homescreen
                file_manager_homescreen = data.Routine(
                    name='file_manager_homescreen',
                    components=[file_manager_mouse_homescreen, file_manager_textbox_homescreen],
                )
                file_manager_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from file_manager_code_homescreen
                task_bar.allocate_target(index=0)
                task_bar.show()
                
                serial_connector.write(SerialConnector.FILE_MANAGER_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the file_manager_mouse_homescreen
                file_manager_mouse_homescreen.x = []
                file_manager_mouse_homescreen.y = []
                file_manager_mouse_homescreen.leftButton = []
                file_manager_mouse_homescreen.midButton = []
                file_manager_mouse_homescreen.rightButton = []
                file_manager_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                file_manager_mouse_homescreen.mouseClock.reset()
                file_manager_textbox_homescreen.reset()
                file_manager_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                file_manager_textbox_homescreen.setSize((2.0, FONT_SIZE))
                file_manager_textbox_homescreen.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
                # store start times for file_manager_homescreen
                file_manager_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                file_manager_homescreen.tStart = globalClock.getTime(format='float')
                file_manager_homescreen.status = STARTED
                thisExp.addData('file_manager_homescreen.started', file_manager_homescreen.tStart)
                file_manager_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                file_manager_homescreenComponents = file_manager_homescreen.components
                for thisComponent in file_manager_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "file_manager_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(file_manager_opening_task, data.TrialHandler2) and thisFile_manager_opening_task.thisN != file_manager_opening_task.thisTrial.thisN:
                    continueRoutine = False
                file_manager_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from file_manager_code_homescreen
                    continueRoutine = task_bar.update()
                    # *file_manager_mouse_homescreen* updates
                    
                    # if file_manager_mouse_homescreen is starting this frame...
                    if file_manager_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        file_manager_mouse_homescreen.frameNStart = frameN  # exact frame index
                        file_manager_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        file_manager_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(file_manager_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'file_manager_mouse_homescreen.started')
                        # update status
                        file_manager_mouse_homescreen.status = STARTED
                        prevButtonState = file_manager_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    if file_manager_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = file_manager_mouse_homescreen.getPos()
                        file_manager_mouse_homescreen.x.append(x)
                        file_manager_mouse_homescreen.y.append(y)
                        buttons = file_manager_mouse_homescreen.getPressed()
                        file_manager_mouse_homescreen.leftButton.append(buttons[0])
                        file_manager_mouse_homescreen.midButton.append(buttons[1])
                        file_manager_mouse_homescreen.rightButton.append(buttons[2])
                        file_manager_mouse_homescreen.time.append(file_manager_mouse_homescreen.mouseClock.getTime())
                    
                    # *file_manager_textbox_homescreen* updates
                    
                    # if file_manager_textbox_homescreen is starting this frame...
                    if file_manager_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        file_manager_textbox_homescreen.frameNStart = frameN  # exact frame index
                        file_manager_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        file_manager_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(file_manager_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        file_manager_textbox_homescreen.status = STARTED
                        file_manager_textbox_homescreen.setAutoDraw(True)
                    
                    # if file_manager_textbox_homescreen is active this frame...
                    if file_manager_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        file_manager_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in file_manager_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "file_manager_homescreen" ---
                for thisComponent in file_manager_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for file_manager_homescreen
                file_manager_homescreen.tStop = globalClock.getTime(format='float')
                file_manager_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('file_manager_homescreen.stopped', file_manager_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from file_manager_code_homescreen
                
                #serial_connector.write(SerialConnector.FILE_MANAGER_HOMESCREEN_END)
                # store data for file_manager_opening_task (TrialHandler)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.x', file_manager_mouse_homescreen.x)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.y', file_manager_mouse_homescreen.y)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.leftButton', file_manager_mouse_homescreen.leftButton)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.midButton', file_manager_mouse_homescreen.midButton)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.rightButton', file_manager_mouse_homescreen.rightButton)
                file_manager_opening_task.addData('file_manager_mouse_homescreen.time', file_manager_mouse_homescreen.time)
                # the Routine "file_manager_homescreen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # set up handler to look after randomisation of conditions etc
                file_opening = data.TrialHandler2(
                    name='file_opening',
                    nReps=30.0, 
                    method='sequential', 
                    extraInfo=expInfo, 
                    originPath=-1, 
                    trialList=[None], 
                    seed=None, 
                )
                thisExp.addLoop(file_opening)  # add the loop to the experiment
                thisFile_opening = file_opening.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisFile_opening.rgb)
                if thisFile_opening != None:
                    for paramName in thisFile_opening:
                        globals()[paramName] = thisFile_opening[paramName]
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                
                for thisFile_opening in file_opening:
                    currentLoop = file_opening
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                    # abbreviate parameter names if possible (e.g. rgb = thisFile_opening.rgb)
                    if thisFile_opening != None:
                        for paramName in thisFile_opening:
                            globals()[paramName] = thisFile_opening[paramName]
                    
                    # --- Prepare to start Routine "file_manager_opening" ---
                    # create an object to store info about Routine file_manager_opening
                    file_manager_opening = data.Routine(
                        name='file_manager_opening',
                        components=[file_manager_opening_textbox, stimuli_reaction_mouse_movement, stimuli_reaction_stimuli_image, stimuli_reaction_mouse_debug],
                    )
                    file_manager_opening.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from file_manager_opening_code
                    base_window.update_title(task_bar.target_name)
                    base_window.show()
                    task_bar.show()
                    
                    # Define size and location of the stimuli
                    size = 0.1
                    radius = 0.4
                    loc_x, loc_y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)
                    
                    # Log clicked elements
                    clicked_elements = 0
                    last_clicked = False
                    
                    serial_connector.write(SerialConnector.FILE_MANAGER_OPENING_BEGIN)
                    file_manager_opening_textbox.reset()
                    file_manager_opening_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                    file_manager_opening_textbox.setSize((2.0, FONT_SIZE))
                    file_manager_opening_textbox.setText("Please double-click to open the folders".format())
                    # setup some python lists for storing info about the stimuli_reaction_mouse_movement
                    stimuli_reaction_mouse_movement.x = []
                    stimuli_reaction_mouse_movement.y = []
                    stimuli_reaction_mouse_movement.leftButton = []
                    stimuli_reaction_mouse_movement.midButton = []
                    stimuli_reaction_mouse_movement.rightButton = []
                    stimuli_reaction_mouse_movement.time = []
                    gotValidClick = False  # until a click is received
                    stimuli_reaction_mouse_movement.mouseClock.reset()
                    stimuli_reaction_stimuli_image.setPos((loc_x, loc_y))
                    stimuli_reaction_stimuli_image.setSize((size, size * BaseWindow.ASPECT_RATIO))
                    # store start times for file_manager_opening
                    file_manager_opening.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    file_manager_opening.tStart = globalClock.getTime(format='float')
                    file_manager_opening.status = STARTED
                    thisExp.addData('file_manager_opening.started', file_manager_opening.tStart)
                    file_manager_opening.maxDuration = None
                    win.color = 'white'
                    win.colorSpace = 'rgb'
                    win.backgroundImage = ''
                    win.backgroundFit = 'none'
                    # keep track of which components have finished
                    file_manager_openingComponents = file_manager_opening.components
                    for thisComponent in file_manager_opening.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "file_manager_opening" ---
                    # if trial has changed, end Routine now
                    if isinstance(file_opening, data.TrialHandler2) and thisFile_opening.thisN != file_opening.thisTrial.thisN:
                        continueRoutine = False
                    file_manager_opening.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from file_manager_opening_code
                        task_bar.update()
                        base_window.update()
                        
                        if not last_clicked and stimuli_reaction_mouse_movement.isPressedIn(stimuli_reaction_stimuli_image):
                            clicked_elements += 1
                            last_clicked = True
                        if not stimuli_reaction_mouse_movement.isPressedIn(stimuli_reaction_stimuli_image):
                            last_clicked = False
                        if clicked_elements >= 2:
                            continueRoutine = False
                        
                        # *file_manager_opening_textbox* updates
                        
                        # if file_manager_opening_textbox is starting this frame...
                        if file_manager_opening_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            file_manager_opening_textbox.frameNStart = frameN  # exact frame index
                            file_manager_opening_textbox.tStart = t  # local t and not account for scr refresh
                            file_manager_opening_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(file_manager_opening_textbox, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            file_manager_opening_textbox.status = STARTED
                            file_manager_opening_textbox.setAutoDraw(True)
                        
                        # if file_manager_opening_textbox is active this frame...
                        if file_manager_opening_textbox.status == STARTED:
                            # update params
                            pass
                        # *stimuli_reaction_mouse_movement* updates
                        
                        # if stimuli_reaction_mouse_movement is starting this frame...
                        if stimuli_reaction_mouse_movement.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_reaction_mouse_movement.frameNStart = frameN  # exact frame index
                            stimuli_reaction_mouse_movement.tStart = t  # local t and not account for scr refresh
                            stimuli_reaction_mouse_movement.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_reaction_mouse_movement, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_reaction_mouse_movement.started')
                            # update status
                            stimuli_reaction_mouse_movement.status = STARTED
                            prevButtonState = stimuli_reaction_mouse_movement.getPressed()  # if button is down already this ISN'T a new click
                        if stimuli_reaction_mouse_movement.status == STARTED:  # only update if started and not finished!
                            x, y = stimuli_reaction_mouse_movement.getPos()
                            stimuli_reaction_mouse_movement.x.append(x)
                            stimuli_reaction_mouse_movement.y.append(y)
                            buttons = stimuli_reaction_mouse_movement.getPressed()
                            stimuli_reaction_mouse_movement.leftButton.append(buttons[0])
                            stimuli_reaction_mouse_movement.midButton.append(buttons[1])
                            stimuli_reaction_mouse_movement.rightButton.append(buttons[2])
                            stimuli_reaction_mouse_movement.time.append(stimuli_reaction_mouse_movement.mouseClock.getTime())
                        
                        # *stimuli_reaction_stimuli_image* updates
                        
                        # if stimuli_reaction_stimuli_image is starting this frame...
                        if stimuli_reaction_stimuli_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_reaction_stimuli_image.frameNStart = frameN  # exact frame index
                            stimuli_reaction_stimuli_image.tStart = t  # local t and not account for scr refresh
                            stimuli_reaction_stimuli_image.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_reaction_stimuli_image, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_reaction_stimuli_image.started')
                            # update status
                            stimuli_reaction_stimuli_image.status = STARTED
                            stimuli_reaction_stimuli_image.setAutoDraw(True)
                        
                        # if stimuli_reaction_stimuli_image is active this frame...
                        if stimuli_reaction_stimuli_image.status == STARTED:
                            # update params
                            pass
                        
                        # *stimuli_reaction_mouse_debug* updates
                        
                        # if stimuli_reaction_mouse_debug is starting this frame...
                        if stimuli_reaction_mouse_debug.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            stimuli_reaction_mouse_debug.frameNStart = frameN  # exact frame index
                            stimuli_reaction_mouse_debug.tStart = t  # local t and not account for scr refresh
                            stimuli_reaction_mouse_debug.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(stimuli_reaction_mouse_debug, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'stimuli_reaction_mouse_debug.started')
                            # update status
                            stimuli_reaction_mouse_debug.status = STARTED
                            stimuli_reaction_mouse_debug.setAutoDraw(True)
                        
                        # if stimuli_reaction_mouse_debug is active this frame...
                        if stimuli_reaction_mouse_debug.status == STARTED:
                            # update params
                            stimuli_reaction_mouse_debug.setText("{:.2f}".format(np.linalg.norm(stimuli_reaction_mouse_movement.getPos() - np.array([loc_x, loc_y]))), log=False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            file_manager_opening.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in file_manager_opening.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "file_manager_opening" ---
                    for thisComponent in file_manager_opening.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for file_manager_opening
                    file_manager_opening.tStop = globalClock.getTime(format='float')
                    file_manager_opening.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('file_manager_opening.stopped', file_manager_opening.tStop)
                    setupWindow(expInfo=expInfo, win=win)
                    # Run 'End Routine' code from file_manager_opening_code
                    thisExp.addData('stimuli_reaction_size', size)
                    thisExp.addData('stimuli_reaction_radius', radius)
                    thisExp.addData('stimuli_reaction_stimuli_location', (loc_x, loc_y))
                    #serial_connector.write(SerialConnector.FILE_MANAGER_OPENING_END)
                    # store data for file_opening (TrialHandler)
                    file_opening.addData('stimuli_reaction_mouse_movement.x', stimuli_reaction_mouse_movement.x)
                    file_opening.addData('stimuli_reaction_mouse_movement.y', stimuli_reaction_mouse_movement.y)
                    file_opening.addData('stimuli_reaction_mouse_movement.leftButton', stimuli_reaction_mouse_movement.leftButton)
                    file_opening.addData('stimuli_reaction_mouse_movement.midButton', stimuli_reaction_mouse_movement.midButton)
                    file_opening.addData('stimuli_reaction_mouse_movement.rightButton', stimuli_reaction_mouse_movement.rightButton)
                    file_opening.addData('stimuli_reaction_mouse_movement.time', stimuli_reaction_mouse_movement.time)
                    # the Routine "file_manager_opening" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    thisExp.nextEntry()
                    
                # completed 30.0 repeats of 'file_opening'
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(file_manager_opening_task, data.TrialHandler2) and thisFile_manager_opening_task.thisN != file_manager_opening_task.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for file_manager_opening_task (TrialHandler)
                file_manager_opening_task.addData('window_close_mouse.x', window_close_mouse.x)
                file_manager_opening_task.addData('window_close_mouse.y', window_close_mouse.y)
                file_manager_opening_task.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                file_manager_opening_task.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                file_manager_opening_task.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                file_manager_opening_task.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('FILE_OPENING') repeats of 'file_manager_opening_task'
            
            
            # set up handler to look after randomisation of conditions etc
            trash_bin = data.TrialHandler2(
                name='trash_bin',
                nReps=event_timer.should_show('TRASH_BIN'), 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(trash_bin)  # add the loop to the experiment
            thisTrash_bin = trash_bin.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrash_bin.rgb)
            if thisTrash_bin != None:
                for paramName in thisTrash_bin:
                    globals()[paramName] = thisTrash_bin[paramName]
            
            for thisTrash_bin in trash_bin:
                currentLoop = trash_bin
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisTrash_bin.rgb)
                if thisTrash_bin != None:
                    for paramName in thisTrash_bin:
                        globals()[paramName] = thisTrash_bin[paramName]
                
                # --- Prepare to start Routine "trash_bin_homescreen" ---
                # create an object to store info about Routine trash_bin_homescreen
                trash_bin_homescreen = data.Routine(
                    name='trash_bin_homescreen',
                    components=[trash_bin_mouse_homescreen, trash_bin_textbox_homescreen],
                )
                trash_bin_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from trash_bin_code_homescreen
                task_bar.allocate_target(index=4)
                task_bar.show()
                
                serial_connector.write(SerialConnector.TRASH_BIN_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the trash_bin_mouse_homescreen
                trash_bin_mouse_homescreen.x = []
                trash_bin_mouse_homescreen.y = []
                trash_bin_mouse_homescreen.leftButton = []
                trash_bin_mouse_homescreen.midButton = []
                trash_bin_mouse_homescreen.rightButton = []
                trash_bin_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                trash_bin_mouse_homescreen.mouseClock.reset()
                trash_bin_textbox_homescreen.reset()
                trash_bin_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                trash_bin_textbox_homescreen.setSize((2.0, FONT_SIZE))
                trash_bin_textbox_homescreen.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
                # store start times for trash_bin_homescreen
                trash_bin_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trash_bin_homescreen.tStart = globalClock.getTime(format='float')
                trash_bin_homescreen.status = STARTED
                thisExp.addData('trash_bin_homescreen.started', trash_bin_homescreen.tStart)
                trash_bin_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                trash_bin_homescreenComponents = trash_bin_homescreen.components
                for thisComponent in trash_bin_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trash_bin_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(trash_bin, data.TrialHandler2) and thisTrash_bin.thisN != trash_bin.thisTrial.thisN:
                    continueRoutine = False
                trash_bin_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from trash_bin_code_homescreen
                    continueRoutine = task_bar.update()
                    # *trash_bin_mouse_homescreen* updates
                    
                    # if trash_bin_mouse_homescreen is starting this frame...
                    if trash_bin_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_mouse_homescreen.frameNStart = frameN  # exact frame index
                        trash_bin_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        trash_bin_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trash_bin_mouse_homescreen.started')
                        # update status
                        trash_bin_mouse_homescreen.status = STARTED
                        prevButtonState = trash_bin_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    if trash_bin_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = trash_bin_mouse_homescreen.getPos()
                        trash_bin_mouse_homescreen.x.append(x)
                        trash_bin_mouse_homescreen.y.append(y)
                        buttons = trash_bin_mouse_homescreen.getPressed()
                        trash_bin_mouse_homescreen.leftButton.append(buttons[0])
                        trash_bin_mouse_homescreen.midButton.append(buttons[1])
                        trash_bin_mouse_homescreen.rightButton.append(buttons[2])
                        trash_bin_mouse_homescreen.time.append(trash_bin_mouse_homescreen.mouseClock.getTime())
                    
                    # *trash_bin_textbox_homescreen* updates
                    
                    # if trash_bin_textbox_homescreen is starting this frame...
                    if trash_bin_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_textbox_homescreen.frameNStart = frameN  # exact frame index
                        trash_bin_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        trash_bin_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        trash_bin_textbox_homescreen.status = STARTED
                        trash_bin_textbox_homescreen.setAutoDraw(True)
                    
                    # if trash_bin_textbox_homescreen is active this frame...
                    if trash_bin_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        trash_bin_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trash_bin_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trash_bin_homescreen" ---
                for thisComponent in trash_bin_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for trash_bin_homescreen
                trash_bin_homescreen.tStop = globalClock.getTime(format='float')
                trash_bin_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('trash_bin_homescreen.stopped', trash_bin_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from trash_bin_code_homescreen
                
                #serial_connector.write(SerialConnector.TRASH_BIN_HOMESCREEN_END)
                # store data for trash_bin (TrialHandler)
                trash_bin.addData('trash_bin_mouse_homescreen.x', trash_bin_mouse_homescreen.x)
                trash_bin.addData('trash_bin_mouse_homescreen.y', trash_bin_mouse_homescreen.y)
                trash_bin.addData('trash_bin_mouse_homescreen.leftButton', trash_bin_mouse_homescreen.leftButton)
                trash_bin.addData('trash_bin_mouse_homescreen.midButton', trash_bin_mouse_homescreen.midButton)
                trash_bin.addData('trash_bin_mouse_homescreen.rightButton', trash_bin_mouse_homescreen.rightButton)
                trash_bin.addData('trash_bin_mouse_homescreen.time', trash_bin_mouse_homescreen.time)
                # the Routine "trash_bin_homescreen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "trash_bin_select" ---
                # create an object to store info about Routine trash_bin_select
                trash_bin_select = data.Routine(
                    name='trash_bin_select',
                    components=[trash_bin_select_mouse, trash_bin_select_textbox],
                )
                trash_bin_select.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from trash_bin_select_code
                base_window.update_title(task_bar.target_name)
                base_window.show()
                task_bar.show()
                trash_items_overlay.reset()
                trash_items_overlay.show()
                serial_connector.write(SerialConnector.TRASH_BIN_SELECT_BEGIN)
                # setup some python lists for storing info about the trash_bin_select_mouse
                trash_bin_select_mouse.x = []
                trash_bin_select_mouse.y = []
                trash_bin_select_mouse.leftButton = []
                trash_bin_select_mouse.midButton = []
                trash_bin_select_mouse.rightButton = []
                trash_bin_select_mouse.time = []
                gotValidClick = False  # until a click is received
                trash_bin_select_mouse.mouseClock.reset()
                trash_bin_select_textbox.reset()
                trash_bin_select_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                trash_bin_select_textbox.setSize((2.0, FONT_SIZE))
                trash_bin_select_textbox.setText("Select all items at once".format())
                # store start times for trash_bin_select
                trash_bin_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trash_bin_select.tStart = globalClock.getTime(format='float')
                trash_bin_select.status = STARTED
                thisExp.addData('trash_bin_select.started', trash_bin_select.tStart)
                trash_bin_select.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                trash_bin_selectComponents = trash_bin_select.components
                for thisComponent in trash_bin_select.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trash_bin_select" ---
                # if trial has changed, end Routine now
                if isinstance(trash_bin, data.TrialHandler2) and thisTrash_bin.thisN != trash_bin.thisTrial.thisN:
                    continueRoutine = False
                trash_bin_select.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from trash_bin_select_code
                    
                    base_window.update()
                    task_bar.update()
                    continueRoutine = trash_items_overlay.update()
                    # *trash_bin_select_mouse* updates
                    
                    # if trash_bin_select_mouse is starting this frame...
                    if trash_bin_select_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_select_mouse.frameNStart = frameN  # exact frame index
                        trash_bin_select_mouse.tStart = t  # local t and not account for scr refresh
                        trash_bin_select_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_select_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trash_bin_select_mouse.started')
                        # update status
                        trash_bin_select_mouse.status = STARTED
                        prevButtonState = trash_bin_select_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if trash_bin_select_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = trash_bin_select_mouse.getPos()
                        trash_bin_select_mouse.x.append(x)
                        trash_bin_select_mouse.y.append(y)
                        buttons = trash_bin_select_mouse.getPressed()
                        trash_bin_select_mouse.leftButton.append(buttons[0])
                        trash_bin_select_mouse.midButton.append(buttons[1])
                        trash_bin_select_mouse.rightButton.append(buttons[2])
                        trash_bin_select_mouse.time.append(trash_bin_select_mouse.mouseClock.getTime())
                    
                    # *trash_bin_select_textbox* updates
                    
                    # if trash_bin_select_textbox is starting this frame...
                    if trash_bin_select_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_select_textbox.frameNStart = frameN  # exact frame index
                        trash_bin_select_textbox.tStart = t  # local t and not account for scr refresh
                        trash_bin_select_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_select_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        trash_bin_select_textbox.status = STARTED
                        trash_bin_select_textbox.setAutoDraw(True)
                    
                    # if trash_bin_select_textbox is active this frame...
                    if trash_bin_select_textbox.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        trash_bin_select.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trash_bin_select.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trash_bin_select" ---
                for thisComponent in trash_bin_select.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for trash_bin_select
                trash_bin_select.tStop = globalClock.getTime(format='float')
                trash_bin_select.tStopRefresh = tThisFlipGlobal
                thisExp.addData('trash_bin_select.stopped', trash_bin_select.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from trash_bin_select_code
                trash_items_overlay.log()
                #serial_connector.write(SerialConnector.TRASH_BIN_SELECT_END)
                # store data for trash_bin (TrialHandler)
                trash_bin.addData('trash_bin_select_mouse.x', trash_bin_select_mouse.x)
                trash_bin.addData('trash_bin_select_mouse.y', trash_bin_select_mouse.y)
                trash_bin.addData('trash_bin_select_mouse.leftButton', trash_bin_select_mouse.leftButton)
                trash_bin.addData('trash_bin_select_mouse.midButton', trash_bin_select_mouse.midButton)
                trash_bin.addData('trash_bin_select_mouse.rightButton', trash_bin_select_mouse.rightButton)
                trash_bin.addData('trash_bin_select_mouse.time', trash_bin_select_mouse.time)
                # the Routine "trash_bin_select" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "trash_bin_confirm" ---
                # create an object to store info about Routine trash_bin_confirm
                trash_bin_confirm = data.Routine(
                    name='trash_bin_confirm',
                    components=[trash_bin_confirm_mouse, trash_bin_confirm_textbox],
                )
                trash_bin_confirm.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from trash_bin_confirm_code
                popup_window.show()
                serial_connector.write(SerialConnector.TRASH_BIN_CONFIRM_BEGIN)
                # setup some python lists for storing info about the trash_bin_confirm_mouse
                trash_bin_confirm_mouse.x = []
                trash_bin_confirm_mouse.y = []
                trash_bin_confirm_mouse.leftButton = []
                trash_bin_confirm_mouse.midButton = []
                trash_bin_confirm_mouse.rightButton = []
                trash_bin_confirm_mouse.time = []
                gotValidClick = False  # until a click is received
                trash_bin_confirm_mouse.mouseClock.reset()
                trash_bin_confirm_textbox.reset()
                trash_bin_confirm_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                trash_bin_confirm_textbox.setSize((2.0, FONT_SIZE))
                trash_bin_confirm_textbox.setText("Confirm your action".format())
                # store start times for trash_bin_confirm
                trash_bin_confirm.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                trash_bin_confirm.tStart = globalClock.getTime(format='float')
                trash_bin_confirm.status = STARTED
                thisExp.addData('trash_bin_confirm.started', trash_bin_confirm.tStart)
                trash_bin_confirm.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                trash_bin_confirmComponents = trash_bin_confirm.components
                for thisComponent in trash_bin_confirm.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trash_bin_confirm" ---
                # if trial has changed, end Routine now
                if isinstance(trash_bin, data.TrialHandler2) and thisTrash_bin.thisN != trash_bin.thisTrial.thisN:
                    continueRoutine = False
                trash_bin_confirm.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from trash_bin_confirm_code
                    
                    base_window.update()
                    task_bar.update()
                    continueRoutine = popup_window.update()
                    # *trash_bin_confirm_mouse* updates
                    
                    # if trash_bin_confirm_mouse is starting this frame...
                    if trash_bin_confirm_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_confirm_mouse.frameNStart = frameN  # exact frame index
                        trash_bin_confirm_mouse.tStart = t  # local t and not account for scr refresh
                        trash_bin_confirm_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_confirm_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trash_bin_confirm_mouse.started')
                        # update status
                        trash_bin_confirm_mouse.status = STARTED
                        prevButtonState = trash_bin_confirm_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if trash_bin_confirm_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = trash_bin_confirm_mouse.getPos()
                        trash_bin_confirm_mouse.x.append(x)
                        trash_bin_confirm_mouse.y.append(y)
                        buttons = trash_bin_confirm_mouse.getPressed()
                        trash_bin_confirm_mouse.leftButton.append(buttons[0])
                        trash_bin_confirm_mouse.midButton.append(buttons[1])
                        trash_bin_confirm_mouse.rightButton.append(buttons[2])
                        trash_bin_confirm_mouse.time.append(trash_bin_confirm_mouse.mouseClock.getTime())
                    
                    # *trash_bin_confirm_textbox* updates
                    
                    # if trash_bin_confirm_textbox is starting this frame...
                    if trash_bin_confirm_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        trash_bin_confirm_textbox.frameNStart = frameN  # exact frame index
                        trash_bin_confirm_textbox.tStart = t  # local t and not account for scr refresh
                        trash_bin_confirm_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trash_bin_confirm_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        trash_bin_confirm_textbox.status = STARTED
                        trash_bin_confirm_textbox.setAutoDraw(True)
                    
                    # if trash_bin_confirm_textbox is active this frame...
                    if trash_bin_confirm_textbox.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        trash_bin_confirm.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trash_bin_confirm.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trash_bin_confirm" ---
                for thisComponent in trash_bin_confirm.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for trash_bin_confirm
                trash_bin_confirm.tStop = globalClock.getTime(format='float')
                trash_bin_confirm.tStopRefresh = tThisFlipGlobal
                thisExp.addData('trash_bin_confirm.stopped', trash_bin_confirm.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from trash_bin_confirm_code
                trash_items_overlay.hide()
                popup_window.hide()
                task_bar.set_trash_empty()
                #serial_connector.write(SerialConnector.TRASH_BIN_CONFIRM_END)
                # store data for trash_bin (TrialHandler)
                trash_bin.addData('trash_bin_confirm_mouse.x', trash_bin_confirm_mouse.x)
                trash_bin.addData('trash_bin_confirm_mouse.y', trash_bin_confirm_mouse.y)
                trash_bin.addData('trash_bin_confirm_mouse.leftButton', trash_bin_confirm_mouse.leftButton)
                trash_bin.addData('trash_bin_confirm_mouse.midButton', trash_bin_confirm_mouse.midButton)
                trash_bin.addData('trash_bin_confirm_mouse.rightButton', trash_bin_confirm_mouse.rightButton)
                trash_bin.addData('trash_bin_confirm_mouse.time', trash_bin_confirm_mouse.time)
                # the Routine "trash_bin_confirm" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(trash_bin, data.TrialHandler2) and thisTrash_bin.thisN != trash_bin.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for trash_bin (TrialHandler)
                trash_bin.addData('window_close_mouse.x', window_close_mouse.x)
                trash_bin.addData('window_close_mouse.y', window_close_mouse.y)
                trash_bin.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                trash_bin.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                trash_bin.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                trash_bin.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('TRASH_BIN') repeats of 'trash_bin'
            
            
            # set up handler to look after randomisation of conditions etc
            notes = data.TrialHandler2(
                name='notes',
                nReps=event_timer.should_show('NOTES'), 
                method='random', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(notes)  # add the loop to the experiment
            thisNote = notes.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisNote.rgb)
            if thisNote != None:
                for paramName in thisNote:
                    globals()[paramName] = thisNote[paramName]
            
            for thisNote in notes:
                currentLoop = notes
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisNote.rgb)
                if thisNote != None:
                    for paramName in thisNote:
                        globals()[paramName] = thisNote[paramName]
                
                # --- Prepare to start Routine "notes_homescreen" ---
                # create an object to store info about Routine notes_homescreen
                notes_homescreen = data.Routine(
                    name='notes_homescreen',
                    components=[notes_mouse_homescreen, notes_textbox_homescreen],
                )
                notes_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from notes_code_homescreen
                task_bar.allocate_target(index=3)
                task_bar.show()
                
                serial_connector.write(SerialConnector.NOTES_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the notes_mouse_homescreen
                notes_mouse_homescreen.x = []
                notes_mouse_homescreen.y = []
                notes_mouse_homescreen.leftButton = []
                notes_mouse_homescreen.midButton = []
                notes_mouse_homescreen.rightButton = []
                notes_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                notes_mouse_homescreen.mouseClock.reset()
                notes_textbox_homescreen.reset()
                notes_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                notes_textbox_homescreen.setSize((2.0, FONT_SIZE))
                notes_textbox_homescreen.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
                # store start times for notes_homescreen
                notes_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                notes_homescreen.tStart = globalClock.getTime(format='float')
                notes_homescreen.status = STARTED
                thisExp.addData('notes_homescreen.started', notes_homescreen.tStart)
                notes_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                notes_homescreenComponents = notes_homescreen.components
                for thisComponent in notes_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "notes_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(notes, data.TrialHandler2) and thisNote.thisN != notes.thisTrial.thisN:
                    continueRoutine = False
                notes_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from notes_code_homescreen
                    continueRoutine = task_bar.update()
                    # *notes_mouse_homescreen* updates
                    
                    # if notes_mouse_homescreen is starting this frame...
                    if notes_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        notes_mouse_homescreen.frameNStart = frameN  # exact frame index
                        notes_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        notes_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(notes_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'notes_mouse_homescreen.started')
                        # update status
                        notes_mouse_homescreen.status = STARTED
                        prevButtonState = notes_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    if notes_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = notes_mouse_homescreen.getPos()
                        notes_mouse_homescreen.x.append(x)
                        notes_mouse_homescreen.y.append(y)
                        buttons = notes_mouse_homescreen.getPressed()
                        notes_mouse_homescreen.leftButton.append(buttons[0])
                        notes_mouse_homescreen.midButton.append(buttons[1])
                        notes_mouse_homescreen.rightButton.append(buttons[2])
                        notes_mouse_homescreen.time.append(notes_mouse_homescreen.mouseClock.getTime())
                    
                    # *notes_textbox_homescreen* updates
                    
                    # if notes_textbox_homescreen is starting this frame...
                    if notes_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        notes_textbox_homescreen.frameNStart = frameN  # exact frame index
                        notes_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        notes_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(notes_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        notes_textbox_homescreen.status = STARTED
                        notes_textbox_homescreen.setAutoDraw(True)
                    
                    # if notes_textbox_homescreen is active this frame...
                    if notes_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        notes_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in notes_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "notes_homescreen" ---
                for thisComponent in notes_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for notes_homescreen
                notes_homescreen.tStop = globalClock.getTime(format='float')
                notes_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('notes_homescreen.stopped', notes_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from notes_code_homescreen
                
                #serial_connector.write(SerialConnector.NOTES_HOMESCREEN_END)
                # store data for notes (TrialHandler)
                notes.addData('notes_mouse_homescreen.x', notes_mouse_homescreen.x)
                notes.addData('notes_mouse_homescreen.y', notes_mouse_homescreen.y)
                notes.addData('notes_mouse_homescreen.leftButton', notes_mouse_homescreen.leftButton)
                notes.addData('notes_mouse_homescreen.midButton', notes_mouse_homescreen.midButton)
                notes.addData('notes_mouse_homescreen.rightButton', notes_mouse_homescreen.rightButton)
                notes.addData('notes_mouse_homescreen.time', notes_mouse_homescreen.time)
                # the Routine "notes_homescreen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "notes_repeat" ---
                # create an object to store info about Routine notes_repeat
                notes_repeat = data.Routine(
                    name='notes_repeat',
                    components=[notes_repeat_textbox, notes_repeat_keyboard],
                )
                notes_repeat.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from notes_repeat_code
                base_window.update_title(task_bar.target_name)
                split_view_editor.allocate_target()
                base_window.show()
                task_bar.show()
                split_view_editor.show()
                serial_connector.write(SerialConnector.NOTES_REPEAT_BEGIN)
                notes_repeat_textbox.reset()
                notes_repeat_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                notes_repeat_textbox.setSize((2.0, FONT_SIZE))
                notes_repeat_textbox.setText("Please repeat the content on the right as it is shown on the left pane".format())
                # create starting attributes for notes_repeat_keyboard
                notes_repeat_keyboard.keys = []
                notes_repeat_keyboard.rt = []
                _notes_repeat_keyboard_allKeys = []
                # store start times for notes_repeat
                notes_repeat.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                notes_repeat.tStart = globalClock.getTime(format='float')
                notes_repeat.status = STARTED
                thisExp.addData('notes_repeat.started', notes_repeat.tStart)
                notes_repeat.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                notes_repeatComponents = notes_repeat.components
                for thisComponent in notes_repeat.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "notes_repeat" ---
                # if trial has changed, end Routine now
                if isinstance(notes, data.TrialHandler2) and thisNote.thisN != notes.thisTrial.thisN:
                    continueRoutine = False
                notes_repeat.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from notes_repeat_code
                    task_bar.update()
                    base_window.update()
                    continueRoutine = split_view_editor.update()
                    
                    # *notes_repeat_textbox* updates
                    
                    # if notes_repeat_textbox is starting this frame...
                    if notes_repeat_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        notes_repeat_textbox.frameNStart = frameN  # exact frame index
                        notes_repeat_textbox.tStart = t  # local t and not account for scr refresh
                        notes_repeat_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(notes_repeat_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        notes_repeat_textbox.status = STARTED
                        notes_repeat_textbox.setAutoDraw(True)
                    
                    # if notes_repeat_textbox is active this frame...
                    if notes_repeat_textbox.status == STARTED:
                        # update params
                        pass
                    
                    # *notes_repeat_keyboard* updates
                    
                    # if notes_repeat_keyboard is starting this frame...
                    if notes_repeat_keyboard.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        notes_repeat_keyboard.frameNStart = frameN  # exact frame index
                        notes_repeat_keyboard.tStart = t  # local t and not account for scr refresh
                        notes_repeat_keyboard.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(notes_repeat_keyboard, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        notes_repeat_keyboard.status = STARTED
                        # keyboard checking is just starting
                        notes_repeat_keyboard.clock.reset()  # now t=0
                    if notes_repeat_keyboard.status == STARTED:
                        theseKeys = notes_repeat_keyboard.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=True)
                        _notes_repeat_keyboard_allKeys.extend(theseKeys)
                        if len(_notes_repeat_keyboard_allKeys):
                            notes_repeat_keyboard.keys = [key.name for key in _notes_repeat_keyboard_allKeys]  # storing all keys
                            notes_repeat_keyboard.rt = [key.rt for key in _notes_repeat_keyboard_allKeys]
                            notes_repeat_keyboard.duration = [key.duration for key in _notes_repeat_keyboard_allKeys]
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        notes_repeat.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in notes_repeat.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "notes_repeat" ---
                for thisComponent in notes_repeat.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for notes_repeat
                notes_repeat.tStop = globalClock.getTime(format='float')
                notes_repeat.tStopRefresh = tThisFlipGlobal
                thisExp.addData('notes_repeat.stopped', notes_repeat.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from notes_repeat_code
                split_view_editor.log()
                #print(split_view_editor.notes_repeat_source.text, split_view_editor.notes_repeat_target.text)
                split_view_editor.hide()
                #serial_connector.write(SerialConnector.NOTES_REPEAT_END)
                # check responses
                if notes_repeat_keyboard.keys in ['', [], None]:  # No response was made
                    notes_repeat_keyboard.keys = None
                notes.addData('notes_repeat_keyboard.keys',notes_repeat_keyboard.keys)
                if notes_repeat_keyboard.keys != None:  # we had a response
                    notes.addData('notes_repeat_keyboard.rt', notes_repeat_keyboard.rt)
                    notes.addData('notes_repeat_keyboard.duration', notes_repeat_keyboard.duration)
                # the Routine "notes_repeat" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(notes, data.TrialHandler2) and thisNote.thisN != notes.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for notes (TrialHandler)
                notes.addData('window_close_mouse.x', window_close_mouse.x)
                notes.addData('window_close_mouse.y', window_close_mouse.y)
                notes.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                notes.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                notes.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                notes.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('NOTES') repeats of 'notes'
            
            
            # set up handler to look after randomisation of conditions etc
            browser = data.TrialHandler2(
                name='browser',
                nReps=event_timer.should_show('BROWSER'), 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(browser)  # add the loop to the experiment
            thisBrowser = browser.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisBrowser.rgb)
            if thisBrowser != None:
                for paramName in thisBrowser:
                    globals()[paramName] = thisBrowser[paramName]
            
            for thisBrowser in browser:
                currentLoop = browser
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisBrowser.rgb)
                if thisBrowser != None:
                    for paramName in thisBrowser:
                        globals()[paramName] = thisBrowser[paramName]
                
                # --- Prepare to start Routine "browser_homescreen" ---
                # create an object to store info about Routine browser_homescreen
                browser_homescreen = data.Routine(
                    name='browser_homescreen',
                    components=[browser_mouse_homescreen, browser_textbox_homescreen],
                )
                browser_homescreen.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from browser_code_homescreen
                task_bar.allocate_target(index=1)
                task_bar.show()
                
                serial_connector.write(SerialConnector.BROWSER_HOMESCREEN_BEGIN)
                # setup some python lists for storing info about the browser_mouse_homescreen
                browser_mouse_homescreen.x = []
                browser_mouse_homescreen.y = []
                browser_mouse_homescreen.leftButton = []
                browser_mouse_homescreen.midButton = []
                browser_mouse_homescreen.rightButton = []
                browser_mouse_homescreen.time = []
                gotValidClick = False  # until a click is received
                browser_mouse_homescreen.mouseClock.reset()
                browser_textbox_homescreen.reset()
                browser_textbox_homescreen.setPos((0, 1.0 - FONT_SIZE / 2))
                browser_textbox_homescreen.setSize((2.0, FONT_SIZE))
                browser_textbox_homescreen.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
                # store start times for browser_homescreen
                browser_homescreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                browser_homescreen.tStart = globalClock.getTime(format='float')
                browser_homescreen.status = STARTED
                thisExp.addData('browser_homescreen.started', browser_homescreen.tStart)
                browser_homescreen.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                browser_homescreenComponents = browser_homescreen.components
                for thisComponent in browser_homescreen.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "browser_homescreen" ---
                # if trial has changed, end Routine now
                if isinstance(browser, data.TrialHandler2) and thisBrowser.thisN != browser.thisTrial.thisN:
                    continueRoutine = False
                browser_homescreen.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from browser_code_homescreen
                    continueRoutine = task_bar.update()
                    # *browser_mouse_homescreen* updates
                    
                    # if browser_mouse_homescreen is starting this frame...
                    if browser_mouse_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_mouse_homescreen.frameNStart = frameN  # exact frame index
                        browser_mouse_homescreen.tStart = t  # local t and not account for scr refresh
                        browser_mouse_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_mouse_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'browser_mouse_homescreen.started')
                        # update status
                        browser_mouse_homescreen.status = STARTED
                        prevButtonState = browser_mouse_homescreen.getPressed()  # if button is down already this ISN'T a new click
                    if browser_mouse_homescreen.status == STARTED:  # only update if started and not finished!
                        x, y = browser_mouse_homescreen.getPos()
                        browser_mouse_homescreen.x.append(x)
                        browser_mouse_homescreen.y.append(y)
                        buttons = browser_mouse_homescreen.getPressed()
                        browser_mouse_homescreen.leftButton.append(buttons[0])
                        browser_mouse_homescreen.midButton.append(buttons[1])
                        browser_mouse_homescreen.rightButton.append(buttons[2])
                        browser_mouse_homescreen.time.append(browser_mouse_homescreen.mouseClock.getTime())
                    
                    # *browser_textbox_homescreen* updates
                    
                    # if browser_textbox_homescreen is starting this frame...
                    if browser_textbox_homescreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_textbox_homescreen.frameNStart = frameN  # exact frame index
                        browser_textbox_homescreen.tStart = t  # local t and not account for scr refresh
                        browser_textbox_homescreen.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_textbox_homescreen, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        browser_textbox_homescreen.status = STARTED
                        browser_textbox_homescreen.setAutoDraw(True)
                    
                    # if browser_textbox_homescreen is active this frame...
                    if browser_textbox_homescreen.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        browser_homescreen.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in browser_homescreen.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "browser_homescreen" ---
                for thisComponent in browser_homescreen.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for browser_homescreen
                browser_homescreen.tStop = globalClock.getTime(format='float')
                browser_homescreen.tStopRefresh = tThisFlipGlobal
                thisExp.addData('browser_homescreen.stopped', browser_homescreen.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from browser_code_homescreen
                
                #serial_connector.write(SerialConnector.BROWSER_HOMESCREEN_END)
                # store data for browser (TrialHandler)
                browser.addData('browser_mouse_homescreen.x', browser_mouse_homescreen.x)
                browser.addData('browser_mouse_homescreen.y', browser_mouse_homescreen.y)
                browser.addData('browser_mouse_homescreen.leftButton', browser_mouse_homescreen.leftButton)
                browser.addData('browser_mouse_homescreen.midButton', browser_mouse_homescreen.midButton)
                browser.addData('browser_mouse_homescreen.rightButton', browser_mouse_homescreen.rightButton)
                browser.addData('browser_mouse_homescreen.time', browser_mouse_homescreen.time)
                # the Routine "browser_homescreen" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "browser_navigation" ---
                # create an object to store info about Routine browser_navigation
                browser_navigation = data.Routine(
                    name='browser_navigation',
                    components=[browser_navigation_textbox, browser_navigation_mouse, browser_navigation_user_key_release],
                )
                browser_navigation.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from browser_navigation_code
                base_window.update_title(task_bar.target_name)
                browser_searchbar.allocate_target()
                base_window.show()
                browser_searchbar.show()
                task_bar.show()
                
                win.winHandle.activate()
                serial_connector.write(SerialConnector.BROWSER_NAVIGATION_BEGIN)
                browser_navigation_textbox.reset()
                browser_navigation_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                browser_navigation_textbox.setSize((2.0, FONT_SIZE))
                browser_navigation_textbox.setText("Please enter the url <i>{}</i>".format(browser_searchbar.target_name))
                # setup some python lists for storing info about the browser_navigation_mouse
                browser_navigation_mouse.x = []
                browser_navigation_mouse.y = []
                browser_navigation_mouse.leftButton = []
                browser_navigation_mouse.midButton = []
                browser_navigation_mouse.rightButton = []
                browser_navigation_mouse.time = []
                gotValidClick = False  # until a click is received
                browser_navigation_mouse.mouseClock.reset()
                # create starting attributes for browser_navigation_user_key_release
                browser_navigation_user_key_release.keys = []
                browser_navigation_user_key_release.rt = []
                _browser_navigation_user_key_release_allKeys = []
                # store start times for browser_navigation
                browser_navigation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                browser_navigation.tStart = globalClock.getTime(format='float')
                browser_navigation.status = STARTED
                thisExp.addData('browser_navigation.started', browser_navigation.tStart)
                browser_navigation.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                browser_navigationComponents = browser_navigation.components
                for thisComponent in browser_navigation.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "browser_navigation" ---
                # if trial has changed, end Routine now
                if isinstance(browser, data.TrialHandler2) and thisBrowser.thisN != browser.thisTrial.thisN:
                    continueRoutine = False
                browser_navigation.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from browser_navigation_code
                    base_window.update()
                    task_bar.update()
                    continueRoutine = browser_searchbar.update()
                    
                    
                    # *browser_navigation_textbox* updates
                    
                    # if browser_navigation_textbox is starting this frame...
                    if browser_navigation_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_navigation_textbox.frameNStart = frameN  # exact frame index
                        browser_navigation_textbox.tStart = t  # local t and not account for scr refresh
                        browser_navigation_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_navigation_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        browser_navigation_textbox.status = STARTED
                        browser_navigation_textbox.setAutoDraw(True)
                    
                    # if browser_navigation_textbox is active this frame...
                    if browser_navigation_textbox.status == STARTED:
                        # update params
                        pass
                    # *browser_navigation_mouse* updates
                    
                    # if browser_navigation_mouse is starting this frame...
                    if browser_navigation_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_navigation_mouse.frameNStart = frameN  # exact frame index
                        browser_navigation_mouse.tStart = t  # local t and not account for scr refresh
                        browser_navigation_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_navigation_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'browser_navigation_mouse.started')
                        # update status
                        browser_navigation_mouse.status = STARTED
                        prevButtonState = browser_navigation_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if browser_navigation_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = browser_navigation_mouse.getPos()
                        browser_navigation_mouse.x.append(x)
                        browser_navigation_mouse.y.append(y)
                        buttons = browser_navigation_mouse.getPressed()
                        browser_navigation_mouse.leftButton.append(buttons[0])
                        browser_navigation_mouse.midButton.append(buttons[1])
                        browser_navigation_mouse.rightButton.append(buttons[2])
                        browser_navigation_mouse.time.append(browser_navigation_mouse.mouseClock.getTime())
                    
                    # *browser_navigation_user_key_release* updates
                    
                    # if browser_navigation_user_key_release is starting this frame...
                    if browser_navigation_user_key_release.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_navigation_user_key_release.frameNStart = frameN  # exact frame index
                        browser_navigation_user_key_release.tStart = t  # local t and not account for scr refresh
                        browser_navigation_user_key_release.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_navigation_user_key_release, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        browser_navigation_user_key_release.status = STARTED
                        # keyboard checking is just starting
                        browser_navigation_user_key_release.clock.reset()  # now t=0
                    if browser_navigation_user_key_release.status == STARTED:
                        theseKeys = browser_navigation_user_key_release.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=True)
                        _browser_navigation_user_key_release_allKeys.extend(theseKeys)
                        if len(_browser_navigation_user_key_release_allKeys):
                            browser_navigation_user_key_release.keys = [key.name for key in _browser_navigation_user_key_release_allKeys]  # storing all keys
                            browser_navigation_user_key_release.rt = [key.rt for key in _browser_navigation_user_key_release_allKeys]
                            browser_navigation_user_key_release.duration = [key.duration for key in _browser_navigation_user_key_release_allKeys]
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        browser_navigation.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in browser_navigation.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "browser_navigation" ---
                for thisComponent in browser_navigation.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for browser_navigation
                browser_navigation.tStop = globalClock.getTime(format='float')
                browser_navigation.tStopRefresh = tThisFlipGlobal
                thisExp.addData('browser_navigation.stopped', browser_navigation.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from browser_navigation_code
                browser_searchbar.log()
                #serial_connector.write(SerialConnector.BROWSER_NAVIGATION_END)
                # store data for browser (TrialHandler)
                browser.addData('browser_navigation_mouse.x', browser_navigation_mouse.x)
                browser.addData('browser_navigation_mouse.y', browser_navigation_mouse.y)
                browser.addData('browser_navigation_mouse.leftButton', browser_navigation_mouse.leftButton)
                browser.addData('browser_navigation_mouse.midButton', browser_navigation_mouse.midButton)
                browser.addData('browser_navigation_mouse.rightButton', browser_navigation_mouse.rightButton)
                browser.addData('browser_navigation_mouse.time', browser_navigation_mouse.time)
                # check responses
                if browser_navigation_user_key_release.keys in ['', [], None]:  # No response was made
                    browser_navigation_user_key_release.keys = None
                browser.addData('browser_navigation_user_key_release.keys',browser_navigation_user_key_release.keys)
                if browser_navigation_user_key_release.keys != None:  # we had a response
                    browser.addData('browser_navigation_user_key_release.rt', browser_navigation_user_key_release.rt)
                    browser.addData('browser_navigation_user_key_release.duration', browser_navigation_user_key_release.duration)
                # the Routine "browser_navigation" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "browser_content" ---
                # create an object to store info about Routine browser_content
                browser_content = data.Routine(
                    name='browser_content',
                    components=[browser_textbox, browser_content_mouse],
                )
                browser_content.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from browser_content_code
                form_overlay.reset()
                form_overlay.show()
                serial_connector.write(SerialConnector.BROWSER_CONTENT_BEGIN)
                browser_textbox.reset()
                browser_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                browser_textbox.setSize((2.0, FONT_SIZE))
                browser_textbox.setText("Please complete the form".format())
                # setup some python lists for storing info about the browser_content_mouse
                browser_content_mouse.x = []
                browser_content_mouse.y = []
                browser_content_mouse.leftButton = []
                browser_content_mouse.midButton = []
                browser_content_mouse.rightButton = []
                browser_content_mouse.time = []
                gotValidClick = False  # until a click is received
                browser_content_mouse.mouseClock.reset()
                # store start times for browser_content
                browser_content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                browser_content.tStart = globalClock.getTime(format='float')
                browser_content.status = STARTED
                thisExp.addData('browser_content.started', browser_content.tStart)
                browser_content.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                browser_contentComponents = browser_content.components
                for thisComponent in browser_content.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "browser_content" ---
                # if trial has changed, end Routine now
                if isinstance(browser, data.TrialHandler2) and thisBrowser.thisN != browser.thisTrial.thisN:
                    continueRoutine = False
                browser_content.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from browser_content_code
                    base_window.update()
                    task_bar.update()
                    browser_searchbar.update()
                    continueRoutine = form_overlay.update()
                    
                    # *browser_textbox* updates
                    
                    # if browser_textbox is starting this frame...
                    if browser_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_textbox.frameNStart = frameN  # exact frame index
                        browser_textbox.tStart = t  # local t and not account for scr refresh
                        browser_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        browser_textbox.status = STARTED
                        browser_textbox.setAutoDraw(True)
                    
                    # if browser_textbox is active this frame...
                    if browser_textbox.status == STARTED:
                        # update params
                        pass
                    # *browser_content_mouse* updates
                    
                    # if browser_content_mouse is starting this frame...
                    if browser_content_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        browser_content_mouse.frameNStart = frameN  # exact frame index
                        browser_content_mouse.tStart = t  # local t and not account for scr refresh
                        browser_content_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(browser_content_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'browser_content_mouse.started')
                        # update status
                        browser_content_mouse.status = STARTED
                        prevButtonState = browser_content_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if browser_content_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = browser_content_mouse.getPos()
                        browser_content_mouse.x.append(x)
                        browser_content_mouse.y.append(y)
                        buttons = browser_content_mouse.getPressed()
                        browser_content_mouse.leftButton.append(buttons[0])
                        browser_content_mouse.midButton.append(buttons[1])
                        browser_content_mouse.rightButton.append(buttons[2])
                        browser_content_mouse.time.append(browser_content_mouse.mouseClock.getTime())
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        browser_content.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in browser_content.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "browser_content" ---
                for thisComponent in browser_content.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for browser_content
                browser_content.tStop = globalClock.getTime(format='float')
                browser_content.tStopRefresh = tThisFlipGlobal
                thisExp.addData('browser_content.stopped', browser_content.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from browser_content_code
                form_overlay.log()
                form_overlay.hide()
                #serial_connector.write(SerialConnector.BROWSER_CONTENT_END)
                # store data for browser (TrialHandler)
                browser.addData('browser_content_mouse.x', browser_content_mouse.x)
                browser.addData('browser_content_mouse.y', browser_content_mouse.y)
                browser.addData('browser_content_mouse.leftButton', browser_content_mouse.leftButton)
                browser.addData('browser_content_mouse.midButton', browser_content_mouse.midButton)
                browser.addData('browser_content_mouse.rightButton', browser_content_mouse.rightButton)
                browser.addData('browser_content_mouse.time', browser_content_mouse.time)
                # the Routine "browser_content" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "window_close" ---
                # create an object to store info about Routine window_close
                window_close = data.Routine(
                    name='window_close',
                    components=[close_textbox, window_close_mouse, success_image],
                )
                window_close.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from close_code
                base_window.allocate_target()
                base_window.show()
                task_bar.show()
                
                serial_connector.write(SerialConnector.WINDOW_CLOSE_BEGIN)
                close_textbox.reset()
                close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
                close_textbox.setSize((2.0, FONT_SIZE))
                close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
                # setup some python lists for storing info about the window_close_mouse
                window_close_mouse.x = []
                window_close_mouse.y = []
                window_close_mouse.leftButton = []
                window_close_mouse.midButton = []
                window_close_mouse.rightButton = []
                window_close_mouse.time = []
                gotValidClick = False  # until a click is received
                window_close_mouse.mouseClock.reset()
                # store start times for window_close
                window_close.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                window_close.tStart = globalClock.getTime(format='float')
                window_close.status = STARTED
                thisExp.addData('window_close.started', window_close.tStart)
                window_close.maxDuration = None
                win.color = 'white'
                win.colorSpace = 'rgb'
                win.backgroundImage = ''
                win.backgroundFit = 'none'
                # keep track of which components have finished
                window_closeComponents = window_close.components
                for thisComponent in window_close.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "window_close" ---
                # if trial has changed, end Routine now
                if isinstance(browser, data.TrialHandler2) and thisBrowser.thisN != browser.thisTrial.thisN:
                    continueRoutine = False
                window_close.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from close_code
                    task_bar.update()
                    continueRoutine = base_window.update()
                    
                    # *close_textbox* updates
                    
                    # if close_textbox is starting this frame...
                    if close_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        close_textbox.frameNStart = frameN  # exact frame index
                        close_textbox.tStart = t  # local t and not account for scr refresh
                        close_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(close_textbox, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        close_textbox.status = STARTED
                        close_textbox.setAutoDraw(True)
                    
                    # if close_textbox is active this frame...
                    if close_textbox.status == STARTED:
                        # update params
                        pass
                    # *window_close_mouse* updates
                    
                    # if window_close_mouse is starting this frame...
                    if window_close_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        window_close_mouse.frameNStart = frameN  # exact frame index
                        window_close_mouse.tStart = t  # local t and not account for scr refresh
                        window_close_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(window_close_mouse, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'window_close_mouse.started')
                        # update status
                        window_close_mouse.status = STARTED
                        prevButtonState = window_close_mouse.getPressed()  # if button is down already this ISN'T a new click
                    if window_close_mouse.status == STARTED:  # only update if started and not finished!
                        x, y = window_close_mouse.getPos()
                        window_close_mouse.x.append(x)
                        window_close_mouse.y.append(y)
                        buttons = window_close_mouse.getPressed()
                        window_close_mouse.leftButton.append(buttons[0])
                        window_close_mouse.midButton.append(buttons[1])
                        window_close_mouse.rightButton.append(buttons[2])
                        window_close_mouse.time.append(window_close_mouse.mouseClock.getTime())
                    
                    # *success_image* updates
                    
                    # if success_image is starting this frame...
                    if success_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        success_image.frameNStart = frameN  # exact frame index
                        success_image.tStart = t  # local t and not account for scr refresh
                        success_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(success_image, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        success_image.status = STARTED
                        success_image.setAutoDraw(True)
                    
                    # if success_image is active this frame...
                    if success_image.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        window_close.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in window_close.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "window_close" ---
                for thisComponent in window_close.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for window_close
                window_close.tStop = globalClock.getTime(format='float')
                window_close.tStopRefresh = tThisFlipGlobal
                thisExp.addData('window_close.stopped', window_close.tStop)
                setupWindow(expInfo=expInfo, win=win)
                # Run 'End Routine' code from close_code
                base_window.log()
                hide_all()
                
                #serial_connector.write(SerialConnector.WINDOW_CLOSE_END)
                # store data for browser (TrialHandler)
                browser.addData('window_close_mouse.x', window_close_mouse.x)
                browser.addData('window_close_mouse.y', window_close_mouse.y)
                browser.addData('window_close_mouse.leftButton', window_close_mouse.leftButton)
                browser.addData('window_close_mouse.midButton', window_close_mouse.midButton)
                browser.addData('window_close_mouse.rightButton', window_close_mouse.rightButton)
                browser.addData('window_close_mouse.time', window_close_mouse.time)
                # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed event_timer.should_show('BROWSER') repeats of 'browser'
            
            thisExp.nextEntry()
            
        # completed 5.0 repeats of 'tasks'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "loop_end" ---
        # create an object to store info about Routine loop_end
        loop_end = data.Routine(
            name='loop_end',
            components=[],
        )
        loop_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from loop_end_code
        camera_view.hide()
        # store start times for loop_end
        loop_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        loop_end.tStart = globalClock.getTime(format='float')
        loop_end.status = STARTED
        thisExp.addData('loop_end.started', loop_end.tStart)
        loop_end.maxDuration = None
        # keep track of which components have finished
        loop_endComponents = loop_end.components
        for thisComponent in loop_end.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "loop_end" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        loop_end.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                loop_end.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in loop_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "loop_end" ---
        for thisComponent in loop_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for loop_end
        loop_end.tStop = globalClock.getTime(format='float')
        loop_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('loop_end.stopped', loop_end.tStop)
        # Run 'End Routine' code from loop_end_code
        camera_connector.save()
        # the Routine "loop_end" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 60.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "experiment_end" ---
    # create an object to store info about Routine experiment_end
    experiment_end = data.Routine(
        name='experiment_end',
        components=[experiment_end_text, experiment_end_mouse, experiment_end_key],
    )
    experiment_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from experiment_end_code
    serial_connector.write(SerialConnector.EXP_END)
    # setup some python lists for storing info about the experiment_end_mouse
    experiment_end_mouse.x = []
    experiment_end_mouse.y = []
    experiment_end_mouse.leftButton = []
    experiment_end_mouse.midButton = []
    experiment_end_mouse.rightButton = []
    experiment_end_mouse.time = []
    gotValidClick = False  # until a click is received
    experiment_end_mouse.mouseClock.reset()
    # create starting attributes for experiment_end_key
    experiment_end_key.keys = []
    experiment_end_key.rt = []
    _experiment_end_key_allKeys = []
    # store start times for experiment_end
    experiment_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    experiment_end.tStart = globalClock.getTime(format='float')
    experiment_end.status = STARTED
    thisExp.addData('experiment_end.started', experiment_end.tStart)
    experiment_end.maxDuration = None
    # keep track of which components have finished
    experiment_endComponents = experiment_end.components
    for thisComponent in experiment_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "experiment_end" ---
    experiment_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *experiment_end_text* updates
        
        # if experiment_end_text is starting this frame...
        if experiment_end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_end_text.frameNStart = frameN  # exact frame index
            experiment_end_text.tStart = t  # local t and not account for scr refresh
            experiment_end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'experiment_end_text.started')
            # update status
            experiment_end_text.status = STARTED
            experiment_end_text.setAutoDraw(True)
        
        # if experiment_end_text is active this frame...
        if experiment_end_text.status == STARTED:
            # update params
            pass
        # *experiment_end_mouse* updates
        
        # if experiment_end_mouse is starting this frame...
        if experiment_end_mouse.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_end_mouse.frameNStart = frameN  # exact frame index
            experiment_end_mouse.tStart = t  # local t and not account for scr refresh
            experiment_end_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_end_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'experiment_end_mouse.started')
            # update status
            experiment_end_mouse.status = STARTED
            prevButtonState = experiment_end_mouse.getPressed()  # if button is down already this ISN'T a new click
        if experiment_end_mouse.status == STARTED:  # only update if started and not finished!
            buttons = experiment_end_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = experiment_end_mouse.getPos()
                    experiment_end_mouse.x.append(x)
                    experiment_end_mouse.y.append(y)
                    buttons = experiment_end_mouse.getPressed()
                    experiment_end_mouse.leftButton.append(buttons[0])
                    experiment_end_mouse.midButton.append(buttons[1])
                    experiment_end_mouse.rightButton.append(buttons[2])
                    experiment_end_mouse.time.append(experiment_end_mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # *experiment_end_key* updates
        
        # if experiment_end_key is starting this frame...
        if experiment_end_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            experiment_end_key.frameNStart = frameN  # exact frame index
            experiment_end_key.tStart = t  # local t and not account for scr refresh
            experiment_end_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(experiment_end_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('experiment_end_key.started', t)
            # update status
            experiment_end_key.status = STARTED
            # keyboard checking is just starting
            experiment_end_key.clock.reset()  # now t=0
            experiment_end_key.clearEvents(eventType='keyboard')
        if experiment_end_key.status == STARTED:
            theseKeys = experiment_end_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _experiment_end_key_allKeys.extend(theseKeys)
            if len(_experiment_end_key_allKeys):
                experiment_end_key.keys = _experiment_end_key_allKeys[-1].name  # just the last key pressed
                experiment_end_key.rt = _experiment_end_key_allKeys[-1].rt
                experiment_end_key.duration = _experiment_end_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            experiment_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in experiment_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "experiment_end" ---
    for thisComponent in experiment_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for experiment_end
    experiment_end.tStop = globalClock.getTime(format='float')
    experiment_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('experiment_end.stopped', experiment_end.tStop)
    # Run 'End Routine' code from experiment_end_code
    serial_connector.write(SerialConnector.EEG_STOP_RECORDING)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('experiment_end_mouse.x', experiment_end_mouse.x)
    thisExp.addData('experiment_end_mouse.y', experiment_end_mouse.y)
    thisExp.addData('experiment_end_mouse.leftButton', experiment_end_mouse.leftButton)
    thisExp.addData('experiment_end_mouse.midButton', experiment_end_mouse.midButton)
    thisExp.addData('experiment_end_mouse.rightButton', experiment_end_mouse.rightButton)
    thisExp.addData('experiment_end_mouse.time', experiment_end_mouse.time)
    # check responses
    if experiment_end_key.keys in ['', [], None]:  # No response was made
        experiment_end_key.keys = None
    thisExp.addData('experiment_end_key.keys',experiment_end_key.keys)
    if experiment_end_key.keys != None:  # we had a response
        thisExp.addData('experiment_end_key.rt', experiment_end_key.rt)
        thisExp.addData('experiment_end_key.duration', experiment_end_key.duration)
    thisExp.nextEntry()
    # the Routine "experiment_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from global_code
    serial_connector.close()
    camera_connector.close()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(enable=False)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
