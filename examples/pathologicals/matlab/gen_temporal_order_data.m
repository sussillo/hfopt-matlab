function [inputs, outputs, did_change_data, net] = gen_temporal_order_data(net, simparams, nexamples, turnover_percentage, inputs_last, outputs_last)

did_change_data = false;
inputs = {};
outputs = {};


pathos_type = simparams.pathosType;
pathos_subtype = simparams.pathosSubtype;
T = simparams.pathosLength;


inputs_new = cell(1,nexamples);
outputs_new = cell(1,nexamples);
if ( turnover_percentage > 0.0 || isempty(inputs_last ))
    for i = 1:nexamples
        switch pathos_subtype
            case '2bit'
                % These are constants associated with the temporal order problems.
		to1_frac = 1/10;
		to2_frac = 2/10;
		to3_frac = 5/10;
		to4_frac = 6/10;



                ninputs = 6;
                noutputs = 4;
                irr_syms = randi(4, [1 T]) + 2;	 % ... 3,5,4,3,5,6, ...
                rev_syms = randi(2, [1 2]);
		t1 = randi( floor((to2_frac - to1_frac)*T) ) + floor(to1_frac*T);
		t2 = randi( floor((to4_frac - to3_frac)*T) ) + floor(to3_frac*T);
                t3 = t2;
		
	    case '3bit'
	        % These are constants associated with the temporal order problems.
		to1_frac = 1/10;
		to2_frac = 2/10;
		to3_frac = 3/10;
		to4_frac = 4/10;
		to5_frac = 6/10;
		to6_frac = 7/10;

                ninputs = 6;
                noutputs = 8;
                irr_syms = randi(4, [1 T]) + 2;	 % ... 3,5,4,3,5,6, ...
                rev_syms = randi(2, [1 3]);

		t1 = randi( floor((to2_frac - to1_frac)*T) ) + floor(to1_frac*T);
		t2 = randi( floor((to4_frac - to3_frac)*T) ) + floor(to3_frac*T);
		t3 = randi( floor((to6_frac - to5_frac)*T) ) + floor(to5_frac*T);
		
            otherwise
                assert ( false, 'Case not implemented.');
        end
        
        
        % 1-of-K encoding        
        input = zeros(ninputs,T);
        for t = 1:T
            if ( t ~= t1 && t ~= t2 && t ~= t3 )
                input(irr_syms(t),t) = 1;
            end
        end
        input(rev_syms(1), t1) = 1;
        input(rev_syms(2), t2) = 1;
        input(rev_syms(3), t3) = 1;
	
        output = NaN(noutputs,T);
        switch pathos_subtype
            case '2bit'
                output(:,end) = zeros(noutputs,1);
                if rev_syms(1) == 1 && rev_syms(2) == 1
                    output(1,end) = 1;
                elseif rev_syms(1) == 1 && rev_syms(2) == 2
                    output(2,end) = 1;
                elseif rev_syms(1) == 2 && rev_syms(2) == 1
                    output(3,end) = 1;
                elseif rev_syms(1) == 2 && rev_syms(2) == 2
                    output(4,end) = 1;
                else
                    assert( false, 'Something''s wrong.');
                end
                
	 case '3bit'
                output(:,end) = zeros(noutputs,1);
                if rev_syms(1) == 1 && rev_syms(2) == 1 && rev_syms(3) == 1 
                    output(1,end) = 1;
                elseif rev_syms(1) == 1 && rev_syms(2) == 1 && rev_syms(3) == 2
                    output(2,end) = 1;
                elseif rev_syms(1) == 1 && rev_syms(2) == 2 && rev_syms(3) == 1
                    output(3,end) = 1;
                elseif rev_syms(1) == 1 && rev_syms(2) == 2 && rev_syms(3) == 2
                    output(4,end) = 1;
		elseif rev_syms(1) == 2 && rev_syms(2) == 1 && rev_syms(3) == 1 
                    output(5,end) = 1;
                elseif rev_syms(1) == 2 && rev_syms(2) == 1 && rev_syms(3) == 2
                    output(6,end) = 1;
                elseif rev_syms(1) == 2 && rev_syms(2) == 2 && rev_syms(3) == 1
                    output(7,end) = 1;
                elseif rev_syms(1) == 2 && rev_syms(2) == 2 && rev_syms(3) == 2
                    output(8,end) = 1;		    
                else
                    assert( false, 'Something''s wrong.');
                end
	  
            otherwise
                assert (false, 'Case not implemented yet.');
        end
        inputs_new{i} = input;
        outputs_new{i} = output;
    end
end

if ( isempty( inputs_last ) )
    inputs = inputs_new;
    outputs = outputs_new;
    did_change_data = true;
else
    nnew = ceil(nexamples * turnover_percentage);
    
    if nnew > 0
        did_change_data = true;
        inputs = inputs_last;
        outputs = outputs_last;
        
        rpidxs = randperm(nexamples);
        rpidxs = rpidxs(1:nnew);
        
        for i = 1:nnew
            inputs{rpidxs(i)} = inputs_new{i};
            outputs{rpidxs(i)} = outputs_new{i};
        end
    end
end



