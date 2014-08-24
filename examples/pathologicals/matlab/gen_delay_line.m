function [inputs, outputs, did_change_data, net] = gen_delay_line(net, simparams, nexamples, turnover_percentage, inputs_last, outputs_last)

inputs = {};
outputs = {};
did_change_data = false;

if nexamples == 0
    return;
end

pathos_type = simparams.pathosType;
pathos_subtype = simparams.pathosSubtype;

T = simparams.pathosLength;
nintegers = simparams.nIntegers;
mem_sequence_length = simparams.sequenceLength;

inputs_new = cell(1,nexamples);
outputs_new = cell(1,nexamples);

% For the noiseless case, we can pregenerate all the examples, so it's pointless to work with 10,000 examples.
switch pathos_subtype
    case 'noiseless_memorization'
        nallseqs = nintegers^mem_sequence_length;
        mem_sequences_string = dec2base(0:nallseqs-1,nintegers);
                
        mem_sequences = zeros(nexamples, mem_sequence_length);
        rpidxs = randperm(nallseqs);
        for i = 1:nexamples
            rpidx = rpidxs(i);
            for j = 1:mem_sequence_length
                mem_sequences(i,j) = str2double(mem_sequences_string(rpidx,j)) + 1;  % These will be indices, so plus 1.
            end
        end
end


if ( turnover_percentage > 0.0 || isempty(inputs_last ))
    for i = 1:nexamples
        
        switch pathos_subtype
            case 'noiseless_memorization'
                
                ninputs = nintegers + 2;
                wait_value = nintegers+1;
                go_value = nintegers+2;
                msl = mem_sequence_length;
                %mem_sequence = randi(nintegers, [1 mem_sequence_length]); 
                %midx = mod(i, nallseqs) + 1;
                midx = i;
                mem_sequence = mem_sequences(midx,:);
                
                complete_sequence_length = msl + T + 1 + msl;
                input_sequence = [mem_sequence wait_value*ones(1,T) go_value wait_value*ones(1,msl)];
                output_sequence = [wait_value*ones(1,msl) wait_value*ones(1,T) wait_value mem_sequence];
                
                input = zeros(ninputs, complete_sequence_length );
                output = zeros(ninputs, complete_sequence_length );
                for j = 1:complete_sequence_length
                    input(input_sequence(j),j) = 1;
                    output(output_sequence(j),j) = 1;
                end
                
            case 'delayline'
                
                % 	input = zeros(1,pathos_length);
                % 	input(1) = rand(1);
                % 	input(25) = rand(1);
                % 	input(50) = rand(1);
                % 	input(75) = rand(1);
                
                % 	%output = NaN(1,pathos_length);
                % 	output = zeros(1,pathos_length);
                % 	output(25) = input(1);
                % 	output(50) = input(25);
                % 	output(75) = input(50);
                % 	output(end) = input(75);
                
                
                delay = 35;
                input = rand(1,pathos_length);
                output = NaN(1,pathos_length);
                %output(end-delay+1:end) = input(end-2*delay+1:end-delay);
                output(delay+1:end) = input(1:end-delay);
        end
        
        inputs_new{i} = input;
        outputs_new{i} = output;
        
    end
end

if ( isempty( inputs_last ) )
    did_change_data = true;
    inputs = inputs_new;
    outputs = outputs_new;
else
    nnew = ceil(nexamples * turnover_percentage);
    
    inputs = inputs_last;
    outputs = outputs_last;
    
    if ( nnew > 0 )
        did_change_data = true;
        disp(['Turning over ' num2str(nnew) ' examples.']);
    end
    
    rpidxs = randperm(nexamples);
    rpidxs = rpidxs(1:nnew);
    
    for i = 1:nnew
        inputs{rpidxs(i)} = inputs_new{i};
        outputs{rpidxs(i)} = outputs_new{i};
    end
end

%for i = 1:32; subplot(1,2,1); imagesc(inputs{i}); subplot(122); imagesc(targets{i}); pause(0.25); end


