function feats_frames = calc_RAPIQUE_CNN_Saliency(test_video, width, height, ...
                                            framerate, minside, net, layer, log_level)
    feats_frames = [];
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        feats_frames = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    if log_level == 1
        fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
                floor(file_length/width/height/1.5));
    end
    % get frame number
    nb_frames = floor(file_length/width/height/1.5);
    
    % get features for each chunk
    blk_idx = 0;
    for fr = floor(framerate/2):framerate:nb_frames-2
        blk_idx = blk_idx + 1;
        if log_level == 1
        fprintf('Processing %d-th block...\n', blk_idx);
        end
        % read uniformly sampled 3 frames for each 1-sec chunk
        this_YUV_frame = YUVread(test_file,[width height],fr);
        prev_YUV_frame = YUVread(test_file,[width height],max(1,fr-floor(framerate/3)));
        next_YUV_frame = YUVread(test_file,[width height],min(nb_frames-2,fr+floor(framerate/3)));
        this_rgb = ycbcr2rgb(uint8(this_YUV_frame));
    	prev_rgb = ycbcr2rgb(uint8(prev_YUV_frame));
        next_rgb = ycbcr2rgb(uint8(next_YUV_frame));

        % subsample to 512p resolution
        sside = min(size(this_YUV_frame,1), size(this_YUV_frame,2));
        ratio = minside / sside;
        if ratio < 1
            %this_rgb = imresize(this_rgb, ratio);
            prev_rgb = imresize(prev_rgb, ratio);
            next_rgb = imresize(next_rgb, ratio);
        end
        
        feats_per_frame = [];
        %% extract spatial NSS features - 680-dim
        
        %% extract deep learning features
        if log_level == 1
        fprintf('- Extracting CNN features (1 fps) ...')
        end
        input_size = net.Layers(1).InputSize;
        im_scale = imresize(this_rgb, [input_size(1), input_size(2)]);
        if log_level == 1, tic; end
        feats_spt_deep = activations(net, im_scale, layer, ...
                            'ExecutionEnvironment','cpu');
        if log_level == 1, toc; end
        feats_per_frame = [feats_per_frame, squeeze(feats_spt_deep)'];

        %% saliency

        %% extract temporal NSS features - 476-dim
        if log_level == 1, toc; end
        feats_frames(end+1,:) = feats_per_frame;
    end
    fclose(test_file);
end

% Read one frame from YUV file
function YUV = YUVread(f, dim, frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f, dim(1)*dim(2)*1.5*frnum, 'bof');
    
    % Read Y-component
    Y = fread(f, dim(1)*dim(2), 'uchar');
    if length(Y) < dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y = cast(reshape(Y, dim(1), dim(2)), 'double');
    
    % Read U-component
    U = fread(f, dim(1)*dim(2)/4, 'uchar');
    if length(U) < dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U = cast(reshape(U, dim(1)/2, dim(2)/2), 'double');
    U = imresize(U, 2.0);
    
    % Read V-component
    V = fread(f, dim(1)*dim(2)/4, 'uchar');
    if length(V) < dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V = cast(reshape(V, dim(1)/2, dim(2)/2), 'double');
    V = imresize(V, 2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1) = Y';
    YUV(:,:,2) = U';
    YUV(:,:,3) = V';
end
