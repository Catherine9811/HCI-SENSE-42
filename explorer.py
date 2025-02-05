#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on April 07, 2024, at 20:14
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'explorer'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
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
        originPath='C:\\Users\\MartinBai白心宇\\Documents\\Personal\\HCI\\Computer\\explorer.py',
        savePickle=True, saveWideText=False,
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.WARNING)


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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1536, 960], fullscr=True, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color='white', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
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
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='event')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='Pyglet')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
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
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
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
    
    class Stylizer:
        STYLES = ["windows", "mac"]
        predefined_style = None
        current_style = None
        
        def __init__(self, style=None):
            self.predefined_style = style
            self.reset(style)
                
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
            
        def update_title(self, name):
            self.NAME = name
            self.title.text = name
            
        def allocate_target(self, index=None):
            if index is None or index >= len(self.TOOLS[self.style]):
                self.target_name = np.random.choice(["Close", "Minimize"])
            else:
                self.target_name = list(self.TOOLS[self.style].keys())[index]
        
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
        SLIDER_SIZE = 0.06
        SLIDER_WIDTH = 1.2
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
                thisExp.addData(slider_component.name, slider_component.getRating())
            
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
                ["very low", "", "", "", "", "", "very high"],
                ["very low", "", "", "", "", "", "very high"],
                ["very slow", "", "", "", "", "", "very fast"],
                ["perfect", "", "", "", "", "", "failure"],
                ["very low", "", "", "", "", "", "very high"],
                ["very low", "", "", "", "", "", "very high"],
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
                    granularity=0.0,
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
        WORD_LIST = ['backspace', 'comma', 'period', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        CHAR_LIST = [',', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        
        
        def __init__(self):
            super().__init__()
            self.draw_interface()
            self.allocate_target()
            
        def log(self):
            thisExp.addData('browser_navigation_template', self.target_name)
            thisExp.addData('browser_navigation_user_input', self.get_text())
            
        def update(self):
            self.search_bar.text = "".join([c for c in self.search_bar.text if c in self.CHAR_LIST])
            if self.search_bar.text.lower() == self.target_name.lower():
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
                self.target_name = np.random.choice(["www.google.com", "www.bing.com"])
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
                lineSpacing=0.1,
                bold=False,
                italic=False,
                color="black",
                borderWidth=0.0,
                fillColor=None,
                borderColor=None,
                overflow="visible",
                editable=True,
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
                 anchor='center-left', overflow='visible',
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
    
    
    def hide_all():
        task_bar.hide()
        base_window.hide()
        browser_searchbar.hide()
        form_overlay.hide()
        trash_items_overlay.hide()
        popup_window.hide()
        
    def reset_layout():
        stylizer.reset()
        task_bar.reset()
        base_window.reset()
        popup_window.reset()
        
    
    stylizer = Stylizer()
    task_bar = TaskBar(style=stylizer)
    base_window = BaseWindow(style=stylizer)
    browser_searchbar = BrowserSearchbar()
    form_overlay = FormOverlay(random=True)
    trash_items_overlay = TrashItemsOverlay(N=8)
    popup_window = PopupWindow(style=stylizer)
    
    hide_all()
    
    # --- Initialize components for Routine "style_randomizer" ---
    
    # --- Initialize components for Routine "homescreen_file_manager" ---
    homescreen_file_manager_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='homescreen_file_manager_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "file_manager_dragging" ---
    file_manager_dragging_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='file_manager_dragging_textbox',
         depth=-1, autoLog=False,
    )
    stimuli_dragging_mouse_movement = event.Mouse(win=win)
    x, y = [None, None]
    stimuli_dragging_mouse_movement.mouseClock = core.Clock()
    stimuli_dragging_target_image = visual.ImageStim(
        win=win,
        name='stimuli_dragging_target_image', units='norm', 
        image='resources/target_stimuli.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=0.8,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    stimuli_dragging_stimuli_image = visual.ImageStim(
        win=win,
        name='stimuli_dragging_stimuli_image', units='norm', 
        image='resources/drag_stimuli.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=0.9,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "homescreen_browser" ---
    homescreen_browser_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='homescreen_browser_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "browser_navigation" ---
    browser_navigation_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='browser_navigation_textbox',
         depth=-1, autoLog=False,
    )
    browser_navigation_user_key_release = keyboard.Keyboard()
    
    # --- Initialize components for Routine "browser_content" ---
    browser_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
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
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "homescreen_file_manager" ---
    homescreen_file_manager_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='homescreen_file_manager_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "file_manager_opening" ---
    file_manager_opening_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
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
        ori=0.0, pos=[0,0], size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    stimuli_reaction_mouse_debug = visual.TextStim(win=win, name='stimuli_reaction_mouse_debug',
        text='',
        font='Open Sans',
        units='norm', pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='lightgrey', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "homescreen_trash_bin" ---
    homescreen_trash_bin_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='homescreen_trash_bin_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "trash_bin_select" ---
    trash_bin_select_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trash_bin_select_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "trash_bin_confirm" ---
    trash_bin_confirm_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trash_bin_confirm_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "homescreen_notes" ---
    homescreen_notes_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='homescreen_notes_textbox',
         depth=-1, autoLog=False,
    )
    
    # --- Initialize components for Routine "notes_repeat" ---
    notes_repeat_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='notes_repeat_textbox',
         depth=-1, autoLog=False,
    )
    notes_repeat_source = visual.TextBox2(
         win, text='This is the template.\n\nMultiline.', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
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
         depth=-2, autoLog=False,
    )
    notes_repeat_target = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=2.0, speechPoint=None,
         padding=0.01, alignment='top-left',
         anchor='top-left', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='notes_repeat_target',
         depth=-3, autoLog=False,
    )
    notes_repeat_keyboard = keyboard.Keyboard()
    
    # --- Initialize components for Routine "window_close" ---
    close_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=[0,0],units='norm',     letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=0.1, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='close_textbox',
         depth=-1, autoLog=False,
    )
    success_image = visual.ImageStim(
        win=win,
        name='success_image', units='height', 
        image='resources/success.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "definition" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('definition.started', globalClock.getTime())
    # keep track of which components have finished
    definitionComponents = []
    for thisComponent in definitionComponents:
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
    routineForceEnded = not continueRoutine
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
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in definitionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "definition" ---
    for thisComponent in definitionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('definition.stopped', globalClock.getTime())
    # the Routine "definition" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "style_randomizer" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('style_randomizer.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from style_randomizer_code
        reset_layout()
        # keep track of which components have finished
        style_randomizerComponents = []
        for thisComponent in style_randomizerComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in style_randomizerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "style_randomizer" ---
        for thisComponent in style_randomizerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('style_randomizer.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "style_randomizer" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "homescreen_file_manager" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('homescreen_file_manager.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from homescreen_file_manager_code
        task_bar.allocate_target(index=0)
        task_bar.show()
        
        homescreen_file_manager_textbox.reset()
        homescreen_file_manager_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        homescreen_file_manager_textbox.setSize((2.0, FONT_SIZE))
        homescreen_file_manager_textbox.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
        # keep track of which components have finished
        homescreen_file_managerComponents = [homescreen_file_manager_textbox]
        for thisComponent in homescreen_file_managerComponents:
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
        
        # --- Run Routine "homescreen_file_manager" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from homescreen_file_manager_code
            continueRoutine = task_bar.update()
            
            # *homescreen_file_manager_textbox* updates
            
            # if homescreen_file_manager_textbox is starting this frame...
            if homescreen_file_manager_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                homescreen_file_manager_textbox.frameNStart = frameN  # exact frame index
                homescreen_file_manager_textbox.tStart = t  # local t and not account for scr refresh
                homescreen_file_manager_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(homescreen_file_manager_textbox, 'tStartRefresh')  # time at next scr refresh
                # update status
                homescreen_file_manager_textbox.status = STARTED
                homescreen_file_manager_textbox.setAutoDraw(True)
            
            # if homescreen_file_manager_textbox is active this frame...
            if homescreen_file_manager_textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in homescreen_file_managerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "homescreen_file_manager" ---
        for thisComponent in homescreen_file_managerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('homescreen_file_manager.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "homescreen_file_manager" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        file_dragging = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='file_dragging')
        thisExp.addLoop(file_dragging)  # add the loop to the experiment
        thisFile_dragging = file_dragging.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisFile_dragging.rgb)
        if thisFile_dragging != None:
            for paramName in thisFile_dragging:
                globals()[paramName] = thisFile_dragging[paramName]
        
        for thisFile_dragging in file_dragging:
            currentLoop = file_dragging
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisFile_dragging.rgb)
            if thisFile_dragging != None:
                for paramName in thisFile_dragging:
                    globals()[paramName] = thisFile_dragging[paramName]
            
            # --- Prepare to start Routine "file_manager_dragging" ---
            continueRoutine = True
            # update component parameters for each repeat
            win.color = 'white'
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
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
            
            file_manager_dragging_textbox.reset()
            file_manager_dragging_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
            file_manager_dragging_textbox.setSize((2.0, FONT_SIZE))
            file_manager_dragging_textbox.setText("Please drag the folder to the destination".format())
            # setup some python lists for storing info about the stimuli_dragging_mouse_movement
            stimuli_dragging_mouse_movement.x = []
            stimuli_dragging_mouse_movement.y = []
            stimuli_dragging_mouse_movement.leftButton = []
            stimuli_dragging_mouse_movement.midButton = []
            stimuli_dragging_mouse_movement.rightButton = []
            stimuli_dragging_mouse_movement.time = []
            gotValidClick = False  # until a click is received
            stimuli_dragging_mouse_movement.mouseClock.reset()
            stimuli_dragging_target_image.setPos((target_x, target_y))
            stimuli_dragging_target_image.setSize((size, size * BaseWindow.ASPECT_RATIO))
            stimuli_dragging_stimuli_image.setPos((loc_x, loc_y))
            stimuli_dragging_stimuli_image.setSize((size, size * BaseWindow.ASPECT_RATIO))
            # keep track of which components have finished
            file_manager_draggingComponents = [file_manager_dragging_textbox, stimuli_dragging_mouse_movement, stimuli_dragging_target_image, stimuli_dragging_stimuli_image]
            for thisComponent in file_manager_draggingComponents:
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
            routineForceEnded = not continueRoutine
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
                
                buttons = stimuli_dragging_mouse_movement.getPressed()
                if last_clicked_offset is None and stimuli_dragging_mouse_movement.isPressedIn(stimuli_dragging_stimuli_image):
                    last_clicked_offset = stimuli_dragging_stimuli_image.pos - stimuli_dragging_mouse_movement.getPos()
                elif last_clicked_offset is not None and np.sum(buttons) > 0:
                    stimuli_dragging_stimuli_image.setPos(last_clicked_offset + stimuli_dragging_mouse_movement.getPos())
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
                # *stimuli_dragging_mouse_movement* updates
                
                # if stimuli_dragging_mouse_movement is starting this frame...
                if stimuli_dragging_mouse_movement.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    stimuli_dragging_mouse_movement.frameNStart = frameN  # exact frame index
                    stimuli_dragging_mouse_movement.tStart = t  # local t and not account for scr refresh
                    stimuli_dragging_mouse_movement.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(stimuli_dragging_mouse_movement, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimuli_dragging_mouse_movement.started')
                    # update status
                    stimuli_dragging_mouse_movement.status = STARTED
                    prevButtonState = stimuli_dragging_mouse_movement.getPressed()  # if button is down already this ISN'T a new click
                if stimuli_dragging_mouse_movement.status == STARTED:  # only update if started and not finished!
                    x, y = stimuli_dragging_mouse_movement.getPos()
                    stimuli_dragging_mouse_movement.x.append(x)
                    stimuli_dragging_mouse_movement.y.append(y)
                    buttons = stimuli_dragging_mouse_movement.getPressed()
                    stimuli_dragging_mouse_movement.leftButton.append(buttons[0])
                    stimuli_dragging_mouse_movement.midButton.append(buttons[1])
                    stimuli_dragging_mouse_movement.rightButton.append(buttons[2])
                    stimuli_dragging_mouse_movement.time.append(stimuli_dragging_mouse_movement.mouseClock.getTime())
                
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
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in file_manager_draggingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "file_manager_dragging" ---
            for thisComponent in file_manager_draggingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            setupWindow(expInfo=expInfo, win=win)
            # Run 'End Routine' code from file_manager_dragging_code
            thisExp.addData('size', size)
            thisExp.addData('radius', radius)
            thisExp.addData('angle', angle)
            thisExp.addData('distance', distance)
            thisExp.addData('stimuli_location', (loc_x, loc_y))
            thisExp.addData('target_location', (target_x, target_y))
            thisExp.addData('last_clicked_offset', tuple(last_clicked_offset))
            
            # store data for file_dragging (TrialHandler)
            file_dragging.addData('stimuli_dragging_mouse_movement.x', stimuli_dragging_mouse_movement.x)
            file_dragging.addData('stimuli_dragging_mouse_movement.y', stimuli_dragging_mouse_movement.y)
            file_dragging.addData('stimuli_dragging_mouse_movement.leftButton', stimuli_dragging_mouse_movement.leftButton)
            file_dragging.addData('stimuli_dragging_mouse_movement.midButton', stimuli_dragging_mouse_movement.midButton)
            file_dragging.addData('stimuli_dragging_mouse_movement.rightButton', stimuli_dragging_mouse_movement.rightButton)
            file_dragging.addData('stimuli_dragging_mouse_movement.time', stimuli_dragging_mouse_movement.time)
            # the Routine "file_manager_dragging" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'file_dragging'
        
        
        # --- Prepare to start Routine "window_close" ---
        continueRoutine = True
        # update component parameters for each repeat
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from close_code
        base_window.allocate_target()
        base_window.show()
        task_bar.show()
        close_textbox.reset()
        close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        close_textbox.setSize((2.0, FONT_SIZE))
        close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
        # keep track of which components have finished
        window_closeComponents = [close_textbox, success_image]
        for thisComponent in window_closeComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in window_closeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "window_close" ---
        for thisComponent in window_closeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from close_code
        hide_all()
        # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "homescreen_browser" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('homescreen_browser.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from homescreen_browser_code
        task_bar.allocate_target(index=1)
        task_bar.show()
        
        homescreen_browser_textbox.reset()
        homescreen_browser_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        homescreen_browser_textbox.setSize((2.0, FONT_SIZE))
        homescreen_browser_textbox.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
        # keep track of which components have finished
        homescreen_browserComponents = [homescreen_browser_textbox]
        for thisComponent in homescreen_browserComponents:
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
        
        # --- Run Routine "homescreen_browser" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from homescreen_browser_code
            continueRoutine = task_bar.update()
            
            # *homescreen_browser_textbox* updates
            
            # if homescreen_browser_textbox is starting this frame...
            if homescreen_browser_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                homescreen_browser_textbox.frameNStart = frameN  # exact frame index
                homescreen_browser_textbox.tStart = t  # local t and not account for scr refresh
                homescreen_browser_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(homescreen_browser_textbox, 'tStartRefresh')  # time at next scr refresh
                # update status
                homescreen_browser_textbox.status = STARTED
                homescreen_browser_textbox.setAutoDraw(True)
            
            # if homescreen_browser_textbox is active this frame...
            if homescreen_browser_textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in homescreen_browserComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "homescreen_browser" ---
        for thisComponent in homescreen_browserComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('homescreen_browser.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "homescreen_browser" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "browser_navigation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('browser_navigation.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from browser_navigation_code
        base_window.update_title(task_bar.target_name)
        browser_searchbar.allocate_target()
        base_window.show()
        browser_searchbar.show()
        task_bar.show()
        
        win.winHandle.activate()
        browser_navigation_textbox.reset()
        browser_navigation_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        browser_navigation_textbox.setSize((2.0, FONT_SIZE))
        browser_navigation_textbox.setText("Please enter the url <i>{}</i>".format(browser_searchbar.target_name))
        browser_navigation_user_key_release.keys = []
        browser_navigation_user_key_release.rt = []
        _browser_navigation_user_key_release_allKeys = []
        # keep track of which components have finished
        browser_navigationComponents = [browser_navigation_textbox, browser_navigation_user_key_release]
        for thisComponent in browser_navigationComponents:
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
        routineForceEnded = not continueRoutine
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
                browser_navigation_user_key_release.clearEvents(eventType='keyboard')
            if browser_navigation_user_key_release.status == STARTED:
                theseKeys = browser_navigation_user_key_release.getKeys(keyList=[browser_searchbar.WORD_LIST], ignoreKeys=["escape"], waitRelease=True)
                _browser_navigation_user_key_release_allKeys.extend(theseKeys)
                if len(_browser_navigation_user_key_release_allKeys):
                    browser_navigation_user_key_release.keys = [key.name for key in _browser_navigation_user_key_release_allKeys]  # storing all keys
                    browser_navigation_user_key_release.rt = [key.rt for key in _browser_navigation_user_key_release_allKeys]
                    browser_navigation_user_key_release.duration = [key.duration for key in _browser_navigation_user_key_release_allKeys]
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in browser_navigationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "browser_navigation" ---
        for thisComponent in browser_navigationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('browser_navigation.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from browser_navigation_code
        browser_searchbar.log()
        # check responses
        if browser_navigation_user_key_release.keys in ['', [], None]:  # No response was made
            browser_navigation_user_key_release.keys = None
        trials.addData('browser_navigation_user_key_release.keys',browser_navigation_user_key_release.keys)
        if browser_navigation_user_key_release.keys != None:  # we had a response
            trials.addData('browser_navigation_user_key_release.rt', browser_navigation_user_key_release.rt)
            trials.addData('browser_navigation_user_key_release.duration', browser_navigation_user_key_release.duration)
        # the Routine "browser_navigation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "browser_content" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('browser_content.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from browser_content_code
        form_overlay.reset()
        form_overlay.show()
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
        # keep track of which components have finished
        browser_contentComponents = [browser_textbox, browser_content_mouse]
        for thisComponent in browser_contentComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in browser_contentComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "browser_content" ---
        for thisComponent in browser_contentComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('browser_content.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from browser_content_code
        form_overlay.log()
        form_overlay.hide()
        # store data for trials (TrialHandler)
        trials.addData('browser_content_mouse.x', browser_content_mouse.x)
        trials.addData('browser_content_mouse.y', browser_content_mouse.y)
        trials.addData('browser_content_mouse.leftButton', browser_content_mouse.leftButton)
        trials.addData('browser_content_mouse.midButton', browser_content_mouse.midButton)
        trials.addData('browser_content_mouse.rightButton', browser_content_mouse.rightButton)
        trials.addData('browser_content_mouse.time', browser_content_mouse.time)
        # the Routine "browser_content" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "window_close" ---
        continueRoutine = True
        # update component parameters for each repeat
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from close_code
        base_window.allocate_target()
        base_window.show()
        task_bar.show()
        close_textbox.reset()
        close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        close_textbox.setSize((2.0, FONT_SIZE))
        close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
        # keep track of which components have finished
        window_closeComponents = [close_textbox, success_image]
        for thisComponent in window_closeComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in window_closeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "window_close" ---
        for thisComponent in window_closeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from close_code
        hide_all()
        # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "homescreen_file_manager" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('homescreen_file_manager.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from homescreen_file_manager_code
        task_bar.allocate_target(index=0)
        task_bar.show()
        
        homescreen_file_manager_textbox.reset()
        homescreen_file_manager_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        homescreen_file_manager_textbox.setSize((2.0, FONT_SIZE))
        homescreen_file_manager_textbox.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
        # keep track of which components have finished
        homescreen_file_managerComponents = [homescreen_file_manager_textbox]
        for thisComponent in homescreen_file_managerComponents:
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
        
        # --- Run Routine "homescreen_file_manager" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from homescreen_file_manager_code
            continueRoutine = task_bar.update()
            
            # *homescreen_file_manager_textbox* updates
            
            # if homescreen_file_manager_textbox is starting this frame...
            if homescreen_file_manager_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                homescreen_file_manager_textbox.frameNStart = frameN  # exact frame index
                homescreen_file_manager_textbox.tStart = t  # local t and not account for scr refresh
                homescreen_file_manager_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(homescreen_file_manager_textbox, 'tStartRefresh')  # time at next scr refresh
                # update status
                homescreen_file_manager_textbox.status = STARTED
                homescreen_file_manager_textbox.setAutoDraw(True)
            
            # if homescreen_file_manager_textbox is active this frame...
            if homescreen_file_manager_textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in homescreen_file_managerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "homescreen_file_manager" ---
        for thisComponent in homescreen_file_managerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('homescreen_file_manager.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "homescreen_file_manager" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        file_opening = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='file_opening')
        thisExp.addLoop(file_opening)  # add the loop to the experiment
        thisFile_opening = file_opening.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisFile_opening.rgb)
        if thisFile_opening != None:
            for paramName in thisFile_opening:
                globals()[paramName] = thisFile_opening[paramName]
        
        for thisFile_opening in file_opening:
            currentLoop = file_opening
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisFile_opening.rgb)
            if thisFile_opening != None:
                for paramName in thisFile_opening:
                    globals()[paramName] = thisFile_opening[paramName]
            
            # --- Prepare to start Routine "file_manager_opening" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('file_manager_opening.started', globalClock.getTime())
            win.color = 'white'
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
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
            # keep track of which components have finished
            file_manager_openingComponents = [file_manager_opening_textbox, stimuli_reaction_mouse_movement, stimuli_reaction_stimuli_image, stimuli_reaction_mouse_debug]
            for thisComponent in file_manager_openingComponents:
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
            routineForceEnded = not continueRoutine
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
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in file_manager_openingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "file_manager_opening" ---
            for thisComponent in file_manager_openingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('file_manager_opening.stopped', globalClock.getTime())
            setupWindow(expInfo=expInfo, win=win)
            # Run 'End Routine' code from file_manager_opening_code
            thisExp.addData('stimuli_reaction_size', size)
            thisExp.addData('stimuli_reaction_radius', radius)
            thisExp.addData('stimuli_reaction_stimuli_location', (loc_x, loc_y))
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
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'file_opening'
        
        
        # --- Prepare to start Routine "window_close" ---
        continueRoutine = True
        # update component parameters for each repeat
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from close_code
        base_window.allocate_target()
        base_window.show()
        task_bar.show()
        close_textbox.reset()
        close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        close_textbox.setSize((2.0, FONT_SIZE))
        close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
        # keep track of which components have finished
        window_closeComponents = [close_textbox, success_image]
        for thisComponent in window_closeComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in window_closeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "window_close" ---
        for thisComponent in window_closeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from close_code
        hide_all()
        # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "homescreen_trash_bin" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('homescreen_trash_bin.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from homescreen_trash_bin_code
        task_bar.allocate_target(index=3)
        task_bar.show()
        
        homescreen_trash_bin_textbox.reset()
        homescreen_trash_bin_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        homescreen_trash_bin_textbox.setSize((2.0, FONT_SIZE))
        homescreen_trash_bin_textbox.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
        # keep track of which components have finished
        homescreen_trash_binComponents = [homescreen_trash_bin_textbox]
        for thisComponent in homescreen_trash_binComponents:
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
        
        # --- Run Routine "homescreen_trash_bin" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from homescreen_trash_bin_code
            continueRoutine = task_bar.update()
            
            # *homescreen_trash_bin_textbox* updates
            
            # if homescreen_trash_bin_textbox is starting this frame...
            if homescreen_trash_bin_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                homescreen_trash_bin_textbox.frameNStart = frameN  # exact frame index
                homescreen_trash_bin_textbox.tStart = t  # local t and not account for scr refresh
                homescreen_trash_bin_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(homescreen_trash_bin_textbox, 'tStartRefresh')  # time at next scr refresh
                # update status
                homescreen_trash_bin_textbox.status = STARTED
                homescreen_trash_bin_textbox.setAutoDraw(True)
            
            # if homescreen_trash_bin_textbox is active this frame...
            if homescreen_trash_bin_textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in homescreen_trash_binComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "homescreen_trash_bin" ---
        for thisComponent in homescreen_trash_binComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('homescreen_trash_bin.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "homescreen_trash_bin" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trash_bin_select" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trash_bin_select.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from trash_bin_select_code
        base_window.update_title(task_bar.target_name)
        base_window.show()
        task_bar.show()
        trash_items_overlay.reset()
        trash_items_overlay.show()
        trash_bin_select_textbox.reset()
        trash_bin_select_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        trash_bin_select_textbox.setSize((2.0, FONT_SIZE))
        trash_bin_select_textbox.setText("Select all items at once".format())
        # keep track of which components have finished
        trash_bin_selectComponents = [trash_bin_select_textbox]
        for thisComponent in trash_bin_selectComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trash_bin_selectComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trash_bin_select" ---
        for thisComponent in trash_bin_selectComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trash_bin_select.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from trash_bin_select_code
        trash_items_overlay.log()
        # the Routine "trash_bin_select" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trash_bin_confirm" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trash_bin_confirm.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from trash_bin_confirm_code
        popup_window.show()
        trash_bin_confirm_textbox.reset()
        trash_bin_confirm_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        trash_bin_confirm_textbox.setSize((2.0, FONT_SIZE))
        trash_bin_confirm_textbox.setText("Confirm your action".format())
        # keep track of which components have finished
        trash_bin_confirmComponents = [trash_bin_confirm_textbox]
        for thisComponent in trash_bin_confirmComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trash_bin_confirmComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trash_bin_confirm" ---
        for thisComponent in trash_bin_confirmComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trash_bin_confirm.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from trash_bin_confirm_code
        trash_items_overlay.hide()
        popup_window.hide()
        task_bar.set_trash_empty()
        # the Routine "trash_bin_confirm" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "window_close" ---
        continueRoutine = True
        # update component parameters for each repeat
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from close_code
        base_window.allocate_target()
        base_window.show()
        task_bar.show()
        close_textbox.reset()
        close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        close_textbox.setSize((2.0, FONT_SIZE))
        close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
        # keep track of which components have finished
        window_closeComponents = [close_textbox, success_image]
        for thisComponent in window_closeComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in window_closeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "window_close" ---
        for thisComponent in window_closeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from close_code
        hide_all()
        # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "homescreen_notes" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('homescreen_notes.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from homescreen_notes_code
        task_bar.allocate_target(index=2)
        task_bar.show()
        
        homescreen_notes_textbox.reset()
        homescreen_notes_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        homescreen_notes_textbox.setSize((2.0, FONT_SIZE))
        homescreen_notes_textbox.setText("Open <i>{}</i> on the task bar".format(task_bar.target_name))
        # keep track of which components have finished
        homescreen_notesComponents = [homescreen_notes_textbox]
        for thisComponent in homescreen_notesComponents:
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
        
        # --- Run Routine "homescreen_notes" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from homescreen_notes_code
            continueRoutine = task_bar.update()
            
            # *homescreen_notes_textbox* updates
            
            # if homescreen_notes_textbox is starting this frame...
            if homescreen_notes_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                homescreen_notes_textbox.frameNStart = frameN  # exact frame index
                homescreen_notes_textbox.tStart = t  # local t and not account for scr refresh
                homescreen_notes_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(homescreen_notes_textbox, 'tStartRefresh')  # time at next scr refresh
                # update status
                homescreen_notes_textbox.status = STARTED
                homescreen_notes_textbox.setAutoDraw(True)
            
            # if homescreen_notes_textbox is active this frame...
            if homescreen_notes_textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in homescreen_notesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "homescreen_notes" ---
        for thisComponent in homescreen_notesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('homescreen_notes.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "homescreen_notes" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "notes_repeat" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('notes_repeat.started', globalClock.getTime())
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from notes_repeat_code
        base_window.update_title(task_bar.target_name)
        base_window.show()
        task_bar.show()
        
        PADDING_SIZE = 0.05
        notes_repeat_textbox.reset()
        notes_repeat_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        notes_repeat_textbox.setSize((2.0, FONT_SIZE))
        notes_repeat_textbox.setText("Please repeat the content on the right as it is shown on the left pane".format())
        notes_repeat_source.reset()
        notes_repeat_source.setPos((-1 * PADDING_SIZE, 1.0 - base_window.FONT_SIZE - base_window.TOOLBAR_SIZE - PADDING_SIZE))
        notes_repeat_source.setSize((1.0 - 2 * PADDING_SIZE, 2.0 - base_window.FONT_SIZE - base_window.TOOLBAR_SIZE - base_window.BAR_SIZE - 2 * PADDING_SIZE))
        notes_repeat_target.reset()
        notes_repeat_target.setPos((PADDING_SIZE, 1.0 - base_window.FONT_SIZE - base_window.TOOLBAR_SIZE - PADDING_SIZE))
        notes_repeat_target.setSize((1.0 - 2 * PADDING_SIZE, 2.0 - base_window.FONT_SIZE - base_window.TOOLBAR_SIZE - base_window.BAR_SIZE - 2 * PADDING_SIZE))
        notes_repeat_keyboard.keys = []
        notes_repeat_keyboard.rt = []
        _notes_repeat_keyboard_allKeys = []
        # keep track of which components have finished
        notes_repeatComponents = [notes_repeat_textbox, notes_repeat_source, notes_repeat_target, notes_repeat_keyboard]
        for thisComponent in notes_repeatComponents:
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
        routineForceEnded = not continueRoutine
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
            continueRoutine = not (notes_repeat_source.text == notes_repeat_target.text)
            
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
            
            # *notes_repeat_source* updates
            
            # if notes_repeat_source is starting this frame...
            if notes_repeat_source.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                notes_repeat_source.frameNStart = frameN  # exact frame index
                notes_repeat_source.tStart = t  # local t and not account for scr refresh
                notes_repeat_source.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(notes_repeat_source, 'tStartRefresh')  # time at next scr refresh
                # update status
                notes_repeat_source.status = STARTED
                notes_repeat_source.setAutoDraw(True)
            
            # if notes_repeat_source is active this frame...
            if notes_repeat_source.status == STARTED:
                # update params
                pass
            
            # *notes_repeat_target* updates
            
            # if notes_repeat_target is starting this frame...
            if notes_repeat_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                notes_repeat_target.frameNStart = frameN  # exact frame index
                notes_repeat_target.tStart = t  # local t and not account for scr refresh
                notes_repeat_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(notes_repeat_target, 'tStartRefresh')  # time at next scr refresh
                # update status
                notes_repeat_target.status = STARTED
                notes_repeat_target.setAutoDraw(True)
            
            # if notes_repeat_target is active this frame...
            if notes_repeat_target.status == STARTED:
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
                notes_repeat_keyboard.clearEvents(eventType='keyboard')
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in notes_repeatComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "notes_repeat" ---
        for thisComponent in notes_repeatComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('notes_repeat.stopped', globalClock.getTime())
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from notes_repeat_code
        thisExp.addData('source_text', notes_repeat_source.text)
        thisExp.addData('target_text', notes_repeat_target.text)
        print(notes_repeat_source.text, notes_repeat_target.text)
        trials.addData('notes_repeat_target.text',notes_repeat_target.text)
        # check responses
        if notes_repeat_keyboard.keys in ['', [], None]:  # No response was made
            notes_repeat_keyboard.keys = None
        trials.addData('notes_repeat_keyboard.keys',notes_repeat_keyboard.keys)
        if notes_repeat_keyboard.keys != None:  # we had a response
            trials.addData('notes_repeat_keyboard.rt', notes_repeat_keyboard.rt)
            trials.addData('notes_repeat_keyboard.duration', notes_repeat_keyboard.duration)
        # the Routine "notes_repeat" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "window_close" ---
        continueRoutine = True
        # update component parameters for each repeat
        win.color = 'white'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # Run 'Begin Routine' code from close_code
        base_window.allocate_target()
        base_window.show()
        task_bar.show()
        close_textbox.reset()
        close_textbox.setPos((0, 1.0 - FONT_SIZE / 2))
        close_textbox.setSize((2.0, FONT_SIZE))
        close_textbox.setText("Click <i>{}</i> to Exit {}".format(base_window.target_name, task_bar.target_name))
        # keep track of which components have finished
        window_closeComponents = [close_textbox, success_image]
        for thisComponent in window_closeComponents:
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
        routineForceEnded = not continueRoutine
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in window_closeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "window_close" ---
        for thisComponent in window_closeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from close_code
        hide_all()
        # the Routine "window_close" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'trials'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


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
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
