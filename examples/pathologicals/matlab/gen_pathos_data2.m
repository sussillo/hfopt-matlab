function [inputs, targets, did_change_data, net, all_simdata] = gen_pathos_data2(net, v_inputs_T, v_targets_T, simparams, nexamples, all_simdata, do_inputs, do_targets ) % last are do_inputs, do_targets
 
inputs = {};
targets = {};
did_change_data = true;
TvV_T = 1;  % Will break if there are validation trials.


if nexamples == 0
    return;
end

pathos_type = simparams.pathosType;
T = simparams.pathosLength;

% These constants are associated with the addition, multiplication and xor problems.
Tp_frac = 11/10;
amx_frac1 = 1/10;
amx_frac2 = 1/2;

inputs = cell(1,nexamples);
targets = cell(1,nexamples);

if do_inputs
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
        
        assert ( t1 < Tp, 'Fucked you dumbass.' );
        assert ( t2 < Tp, 'Fucked you dumbass.' );
        
        input = zeros(2,Tp);
        input(1,:) = rand(1,Tp);
        input(1,t1) = r1;
        input(1,t2) = r2;
        input(2,t1) = 1.0;
        input(2,t2) = 1.0;
        
        target = NaN(1,Tp);
        
        switch pathos_type
            case 'addition'
                target(end) = (r1 + r2)/2.0;
            case 'multiplication'
                target(end) = (r1 * r2);
            case 'xor'
                target(end) = xor(r1,r2);
            otherwise
                assert (false, 'Case not implemented yet.');
        end
        inputs{i} = input;
        all_simdata{TvV_T}(i).targets = target;
    end
end

if do_inputs && ~do_targets
    targets = cell(1,nexamples);
    for i = 1:nexamples           
        targets{i} = NaN(net.layers(3).nPost, size(inputs{i},2));
    end
end
if do_targets
    targets = cell(1,nexamples);
    for i = 1:nexamples
        targets{i} = all_simdata{TvV_T}(i).targets;
    end
end


