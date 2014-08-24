function [inputs, outputs, did_change_data, net] = gen_pathos_data(net, simparams, nexamples, turnover_percentage, inputs_last, outputs_last)

inputs = {};
outputs = {};
did_change_data = false;

if nexamples == 0
    return;
end

pathos_type = simparams.pathosType;
T = simparams.pathosLength;

% These constants are associated with the addition, multiplication and xor problems.
Tp_frac = 11/10;  
amx_frac1 = 1/10;
amx_frac2 = 1/2;

inputs_new = cell(1,nexamples);
outputs_new = cell(1,nexamples);

if ( turnover_percentage > 0.0 || isempty(inputs_last ))
    for i = 1:nexamples
	Tp = T + randi(floor(Tp_frac*T) - T, 1);
	switch pathos_type
            case 'addition'
                r1 = rand;
                r2 = rand;
            case 'multiplication'
                r1 = rand;
                r2 = rand;
            case 'xor'
                r1 = rand > 0.5;
                r2 = rand > 0.5;
            otherwise
                assert ( false, 'Case not implemented.');
        end


	time1_max = floor(amx_frac1 * Tp);
	time2_max = floor(amx_frac2 * Tp);
        t1 = randi(time1_max);
        t2 = randi(time2_max - time1_max) + time1_max;
        input = zeros(2,Tp);
        input(1,:) = rand(1,Tp);
        input(1,t1) = r1;
        input(1,t2) = r2;
        input(2,t1) = 1.0;
        input(2,t2) = 1.0;
        
        output = NaN(1,Tp);
        
        switch pathos_type
            case 'addition'
                output(end) = (r1 + r2)/2.0;
            case 'multiplication'
                output(end) = (r1 * r2);
            case 'xor'
                output(end) = xor(r1,r2);
            otherwise
                assert (false, 'Case not implemented yet.');
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
    
    rpidxs = randperm(nexamples);
    rpidxs = rpidxs(1:nnew);
    
    if ( nnew > 0 )
	did_change_data = true;
        disp(['Turning over ' num2str(nnew) ' examples.']);
    end
    
    for i = 1:nnew
        inputs(rpidxs(i)) = inputs_new(i);
        outputs(rpidxs(i)) = outputs_new(i);
    end
end



