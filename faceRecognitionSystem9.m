function faceRecognitionSystem9()
    % Create 'myImages' and 'Report' folders if they do not exist
    if ~exist('myImages', 'dir')
        mkdir('myImages');
    end
    if ~exist('Report', 'dir')
        mkdir('Report');
    end
    
    % Create a UIFigure for the main interface
    fig = uifigure('Name', 'Face Recognition System', 'Position', [100, 100, 360, 400], 'Color', [0.95 0.95 0.95]);
    
    % Add a title
    titleText = uilabel(fig, 'Text', 'Face Recognition System', 'FontWeight', 'bold', 'FontSize', 16, 'Position', [30, 350, 300, 30], 'HorizontalAlignment', 'left');
    
    % Add icons for each function
    iconSize = [32, 32];
    iconAdd = imread('add_icon.png');
    iconRemove = imread('remove_icon.png');
    iconRetrain = imread('retrain_icon.png');
    iconStart = imread('start_icon.png');
    iconReport = imread('report_icon.png');
    iconExit = imread('exit_icon.png');
    
    addFaceButton = uibutton(fig, 'Icon', iconAdd, 'Text', 'Add New Face', 'Position', [30, 270, 300, 50], 'ButtonPushedFcn', @(btn,event) addNewFaceCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    removeFaceButton = uibutton(fig, 'Icon', iconRemove, 'Text', 'Remove Face', 'Position', [30, 220, 300, 50], 'ButtonPushedFcn', @(btn,event) removeFaceCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    retrainModelButton = uibutton(fig, 'Icon', iconRetrain, 'Text', 'Retrain Model', 'Position', [30, 170, 300, 50], 'ButtonPushedFcn', @(btn,event) retrainModelCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    startDetectionButton = uibutton(fig, 'Icon', iconStart, 'Text', 'Start Face Detection', 'Position', [30, 120, 300, 50], 'ButtonPushedFcn', @(btn,event) startFaceDetectionCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    generateReportButton = uibutton(fig, 'Icon', iconReport, 'Text', 'Generate Report', 'Position', [30, 70, 300, 50], 'ButtonPushedFcn', @(btn,event) generateReportCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    exitButton = uibutton(fig, 'Icon', iconExit, 'Text', 'Exit', 'Position', [30, 20, 300, 50], 'ButtonPushedFcn', @(btn,event) exitCallback(), 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

    % Global variable to track if the model has been retrained
    global isModelTrained;
    isModelTrained = false;

    % Define callback functions
    function addNewFaceCallback()
        addNewFace();
    end
    
    function removeFaceCallback()
        removeFace();
    end
    
    function retrainModelCallback()
        retrainModel();
        isModelTrained = true;  % Set the flag to true after retraining the model
    end

    function startFaceDetectionCallback()
        if ~isModelTrained
            % If the model hasn't been retrained, prompt the user
            uialert(fig, 'Please retrain the model before starting face detection.', 'Model Not Trained');
        else
            global myNet;
            startFaceDetection(myNet);
        end
    end
    
    function generateReportCallback()
        generateReport();
    end
    
    function exitCallback()
        close(fig);
    end
end

% Define other necessary functions (trainNetworkModel, addNewFace, removeFace, etc.) here

function myNet = trainNetworkModel()
    % Load a pre-trained, deep, convolutional network
    alex = alexnet;
    layers = alex.Layers;

    % Determine the number of unique labels (subjects) in the training data
    numUniqueLabels = countUniqueLabels();

    % Modify the network to use the appropriate number of categories
    layers(23) = fullyConnectedLayer(numUniqueLabels); 
    layers(25) = classificationLayer;

    % Set up our training data
    allImages = imageDatastore('myImages', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames'); 

    % Split the data into training and test sets
    [trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomize');

    % Data Augmentation
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation', [-10, 10], ...
        'RandXTranslation', [-5, 5], ...
        'RandYTranslation', [-5, 5], ...
        'RandXScale', [0.9, 1.1], ...
        'RandYScale', [0.9, 1.1]);

    augmentedTrainingImages = augmentedImageDatastore([227 227], ...
        trainingImages, 'DataAugmentation', imageAugmenter);

    augmentedTestImages = augmentedImageDatastore([227 227], testImages);

    % Resize validation images to match the network input size
    augmentedTestImagesResized = augmentedImageDatastore([227 227], testImages);

    % Re-train the Network with adjusted hyperparameters
    opts = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.0001, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augmentedTestImagesResized, ...
        'ValidationFrequency', 10, ...
        'Verbose', false, ...
        'Plots', 'training-progress', ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 10);

    myNet = trainNetwork(augmentedTrainingImages, layers, opts);

    % Measure network accuracy
    predictedLabels = classify(myNet, augmentedTestImages);
    accuracy = mean(predictedLabels == testImages.Labels);
    disp(['Test accuracy: ', num2str(accuracy)]);
end

function numUniqueLabels = countUniqueLabels()
    % Set up the image data store
    allImages = imageDatastore('myImages', ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    % Get unique labels (subjects) from the data store
    labels = allImages.Labels;
    numUniqueLabels = numel(unique(labels));
end

function retrainModel()
    global myNet;
    myNet = trainNetworkModel();
end

function addNewFace()
    % Create necessary folders if they do not exist
    if ~exist('myImages', 'dir')
        mkdir('myImages');
    end
    if ~exist('Report', 'dir')
        mkdir('Report');
    end

    % Prompt for the name of the new face
    name = inputdlg('Enter the name of the new face:', 'Add New Face');
    if isempty(name)
        return;
    end
    name = lower(name{1});
    
    % Check if the name already exists
    folderName = fullfile('myImages', name);
    if exist(folderName, 'dir')
        msgbox('The face is already registered under that name.', 'Error', 'error');
        return;
    end
    
    % Create a folder for the new face
    mkdir(folderName);

    % Initialize webcam
    cam = webcam;
    faceDetector = vision.CascadeObjectDetector;
    hf = figure('Name', 'Capture New Face', 'NumberTitle', 'off', 'Position', [100, 100, 600, 500]);
    
    % Instructions for the user
    instructionText = uicontrol('Style', 'text', 'String', 'Move your head slowly left to right, up and down to capture your face', ...
              'Position', [10, 20, 580, 40], 'ForegroundColor', 'blue', 'BackgroundColor', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Progress text
    progressText = uicontrol('Style', 'text', 'String', '0% complete', ...
                             'Position', [10, 450, 580, 40], 'ForegroundColor', 'green', 'BackgroundColor', 'white', 'FontSize', 12, 'FontWeight', 'bold');

    % Capture images of the new face
    numCaptured = 0;
    while numCaptured < 20
        img = snapshot(cam);
        bboxes = step(faceDetector, img);
        if ~isempty(bboxes)
            % Capture only the face region
            faceImage = imcrop(img, bboxes(1, :));
            faceImage = imresize(faceImage, [227 227]);
            imwrite(faceImage, fullfile(folderName, sprintf('%s_%d.jpg', name, numCaptured)));
            numCaptured = numCaptured + 1;

            % Update the progress text
            set(progressText, 'String', sprintf('%d%% complete', round((numCaptured / 20) * 100)));

            % Display captured face in the figure
            imshow(faceImage);
            drawnow;
        end
    end
    
    % Clean up
    clear cam;
    close(hf);
end


function removeFace()
    % List available faces (folders)
    facesDir = dir('myImages');
    faces = {facesDir([facesDir.isdir]).name};
    faces = faces(~ismember(faces, {'.', '..'}));
    
    % Prompt for the face to remove
    [selection, ok] = listdlg('PromptString', 'Select a face to remove:', 'SelectionMode', 'single', 'ListString', faces);
    if ok
        faceToRemove = faces{selection};
        % Double confirm from the user
        choice = questdlg(sprintf('Are you sure you want to delete the face for "%s"?', faceToRemove), ...
                          'Confirm Face Removal', ...
                          'Yes', 'No', 'No');
        if strcmp(choice, 'Yes')
            % Remove the folder
            rmdir(fullfile('myImages', faceToRemove), 's');
            
            % Display a clear, visible, and eye-catching message
            msg = sprintf('Face "%s" removed successfully. For accurate recognition, consider retraining the model.', faceToRemove);
            msgbox(msg, 'Face Removed', 'modal', 'warn');
        end
    end
end

function startFaceDetection(net)
    % Initialize webcam
    cam = webcam;
    % Initialize face detector
    faceDetector = vision.CascadeObjectDetector;

    % Initialize detected faces array
    global detectedFaces;
    detectedFaces = struct('Image', {}, 'BoundingBox', {}, 'Label', {}, 'Confidence', {});

    % Variable to control the loop
    keepRunning = true;

    % Create figure and add exit button
    hf = figure('Name', 'Real-time Image Classification', 'NumberTitle', 'off', ...
                'KeyPressFcn', @(~, event) exitKeyPress(event), ...
                'CloseRequestFcn', @(src, event) closeFigureCallback(src, event));
    uicontrol('Style', 'pushbutton', 'String', 'Exit', 'Position', [20, 20, 60, 30], ...
              'Callback', @(~,~) exitButtonCallback());

    % Key press callback function
    function exitKeyPress(event)
        if strcmp(event.Key, 'q')
            keepRunning = false;
            close(hf);
        end
    end

    % Exit button callback function
    function exitButtonCallback()
        keepRunning = false;
        close(hf);
    end

    % Figure close callback function
    function closeFigureCallback(src, event)
        keepRunning = false;
        delete(src);
    end

    % Main loop for real-time image classification
    while keepRunning
        % Acquire a single image
        rgbImage = snapshot(cam);
        % Detect faces
        bboxes = step(faceDetector, rgbImage);
        
        % Display the image
        imshow(rgbImage);
        hold on;
        
        % Process each detected face
        for i = 1:size(bboxes, 1)
            % Extract and resize the face region to fit AlexNet dimensions
            faceImage = imcrop(rgbImage, bboxes(i, :));
            resizedImage = imresize(faceImage, [227 227]);
            % Classify Image
            [label, scores] = classify(net, resizedImage);
            % Get confidence percentage
            confidence = max(scores) * 100;

            % Determine color based on confidence level
            if confidence < 50
                boxColor = 'r';  % Red for low confidence
                titleColor = 'r';  % Red for low confidence
            else
                boxColor = 'g';  % Green for high confidence
                titleColor = 'g';  % Green for high confidence
            end
            
            % Draw bounding box around detected face
            rectangle('Position', bboxes(i, :), 'EdgeColor', boxColor, 'LineWidth', 2, 'LineStyle', '--');
            
            % Display label with confidence percentage
            titleText = sprintf('Label: %s (%.2f%%)', char(label), confidence);
            text(bboxes(i, 1), bboxes(i, 2) - 10, titleText, 'FontSize', 14, 'Color', titleColor, 'BackgroundColor', 'k', 'Margin', 5);

            % Store detected face
            detectedFaces(end+1) = struct('Image', rgbImage, 'BoundingBox', bboxes(i, :), 'Label', char(label), 'Confidence', confidence);
        end
        
        % Update figure
        hold off;
        drawnow;
    end

    % Clear camera object
    clear('cam');
end

function generateReport()
    global detectedFaces;
    if isempty(detectedFaces)
        msgbox('No faces detected.', 'Generate Report');
        return;
    end

    % Get current date and time
    currentTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');

    % Get unique labels (names)
    labels = unique({detectedFaces.Label});

    % Prepare report data
    reportData = cell(numel(labels), 4);
    for i = 1:numel(labels)
        labelFaces = detectedFaces(strcmp({detectedFaces.Label}, labels{i}));
        confidenceValues = [labelFaces.Confidence];
        avgConfidence = mean(confidenceValues);
        frequency = numel(labelFaces);
        reportData{i, 1} = labels{i};
        reportData{i, 2} = num2str(avgConfidence, '%.2f');
        reportData{i, 3} = num2str(frequency);
        reportData{i, 4} = datestr(currentTime, 'HH:MM:SS');
    end

    % Generate report file name with timestamp
    reportFileName = ['Report_' datestr(currentTime, 'yyyy-mm-dd_HH-MM-SS') '.txt'];
    reportFilePath = fullfile('Report', reportFileName);

    try
        % Open report file for writing
        fileID = fopen(reportFilePath, 'w');
        if fileID == -1
            error('Failed to create report file.');
        end
        
        % Write report header
        fprintf(fileID, 'Face Detection Report\n\n');
        fprintf(fileID, 'Name,Average Confidence,Frequency,Timestamp\n');
        
        % Write report data
        for i = 1:size(reportData, 1)
            fprintf(fileID, '%s,%s,%s,%s\n', reportData{i, :});
        end
        
        % Close the report file
        fclose(fileID);
        
        % Display success message
        msgbox(sprintf('Report generated successfully:\n%s', reportFilePath), 'Generate Report');
    catch ME
        % Close the report file if it was opened
        if exist('fileID', 'var') && fileID ~= -1
            fclose(fileID);
        end
        % Display error message
        errordlg(['Error in generating report: ' ME.message], 'Report Generation Error');
    end
end
