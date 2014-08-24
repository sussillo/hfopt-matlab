function [n_Wru_v, n_Wrr_n, m_Wzr_n, n_WruM_v, n_WrrM_n, m_WzrM_n] = make_subpools(conn_struct, varargin)

do_debug = false;
do_plot = false;

optargin = size(varargin,2);
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'doplot'		
            do_plot = varargin{i+1};
        otherwise
            assert ( false, ['Option ' varargin{i} ' not recognized.']);
    end
end
            

% Note that each input and each output is treated as it's own pool!  This is the way to use the same code in three different conditions and allow the
% flexibility to set different inputs to different recurrent pools and train different outputs from differnt recurrent subpools.

if do_debug
    % for the matrix convention, it's output <- input, so a mat of size (number outputs) x (number inputs).
    rnet_size = 100;
    cs.V = 0;  % Number of inputs
    npools = 4;
    ps = rnet_size / npools;  % Number of neurons in the pools in recurrent network
    cs.Ns = repmat(ps, 1, npools);
    cs.M = 4;
    cs.RUSGraph = []';  % Scale
    cs.RUCGraph = []';  % Connectivity
    cs.RUMGraph = []';  % Modification
    cs.RRSGraph = [1.2 0 0 0; 0 1.2 0 0; 0 0 1.2 0; 0 0 0 1.2];
    cs.RRCGraph = [ps 0 0 0; 0 ps 0 0; 0 0 ps 0; 0 0 0 ps];  % This is the number of connections from one pool to another.
    cs.RRMGraph = [1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1];
    cs.ZRSGraph = [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0];
    cs.ZRCGraph = [ps 0 0 0; 0 ps 0 0; 0 0 ps 0; 0 0 0 ps];
    cs.ZRMGraph = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1 ];
    % These are [ inputs recurrents outputs]
    cs.modMaskFollowsSparseness = [0 0 0];  % All 1s means modification of sparse inits OK, else everything is OK to modify I2R R2R Z2R
    cs.doSparsify = [0 0 0];  % Literally make into a sparse matrix.
   
end
    
I2R = 1;
R2R = 2;
R2Z = 3;
PRE = 1;
POST = 2;

V = conn_struct.V;
M = conn_struct.M;
Nsc{I2R}{PRE} = ones(1,V);
Nsc{I2R}{POST} = conn_struct.Ns;
Nsc{R2R}{PRE} = conn_struct.Ns;
Nsc{R2R}{POST} = conn_struct.Ns;
Nsc{R2Z}{PRE} = conn_struct.Ns;
Nsc{R2Z}{POST} = ones(1,M);



Nc{I2R}{PRE} = V;
Nc{I2R}{POST} = sum(conn_struct.Ns);
Nc{R2R}{PRE} = sum(conn_struct.Ns);
Nc{R2R}{POST} = sum(conn_struct.Ns);
Nc{R2Z}{PRE} = sum(conn_struct.Ns);
Nc{R2Z}{POST} = M;


postpre_sgraph{I2R} = conn_struct.RUSGraph;
postpre_sgraph{R2R} = conn_struct.RRSGraph;
postpre_sgraph{R2Z} = conn_struct.ZRSGraph;

postpre_Cgraph{I2R} = conn_struct.RUCGraph;
postpre_Cgraph{R2R} = conn_struct.RRCGraph;
postpre_Cgraph{R2Z} = conn_struct.ZRCGraph;

postpre_Mgraph{I2R} = conn_struct.RUMGraph;
postpre_Mgraph{R2R} = conn_struct.RRMGraph;
postpre_Mgraph{R2Z} = conn_struct.ZRMGraph;

% if V > 0 
%     npoolsc{I2R}{PRE} = 1;
% else
%     npoolsc{I2R}{PRE} = 0;
% end
npoolsc{I2R}{PRE} = V;

npoolsc{I2R}{POST} = size(postpre_sgraph{R2R},1);
npoolsc{R2R}{PRE} = size(postpre_sgraph{R2R},1); 
npoolsc{R2R}{POST} = size(postpre_sgraph{R2R},1); 
npoolsc{R2Z}{PRE} = size(postpre_sgraph{R2R},1);
% if M > 0 
%     npoolsc{R2Z}{POST} = 1;
% else
%     npoolsc{R2Z}{POST} = 0;
% end
npoolsc{R2Z}{POST} = M;

RU_borders = cumsum(Nsc{I2R}{PRE});
RR_borders = cumsum(conn_struct.Ns);
ZR_borders = cumsum(Nsc{R2Z}{POST});

border_startsc{I2R}{PRE} = [0 RU_borders] + 1;
border_endsc{I2R}{PRE} = RU_borders;
border_startsc{I2R}{POST} = [0 RR_borders] + 1;
border_endsc{I2R}{POST} = RR_borders;

border_startsc{R2R}{PRE} = [0 RR_borders] + 1;
border_endsc{R2R}{PRE} = RR_borders;
border_startsc{R2R}{POST} = [0 RR_borders] + 1;
border_endsc{R2R}{POST} = RR_borders;

border_startsc{R2Z}{PRE} = [0 RR_borders] + 1;
border_endsc{R2Z}{PRE} = RR_borders;
border_startsc{R2Z}{POST} = [0 ZR_borders] + 1;
border_endsc{R2Z}{POST} = ZR_borders;

% Asserts... a lot of them.


for i = [I2R R2R R2Z]
    
    npools_pre = npoolsc{i}{PRE};
    npools_post = npoolsc{i}{POST};
    Ns_pre = Nsc{i}{PRE};
    Ns_post = Nsc{i}{POST};
    border_starts_pre = border_startsc{i}{PRE};
    border_ends_pre = border_endsc{i}{PRE};
    border_starts_post = border_startsc{i}{POST};
    border_ends_post = border_endsc{i}{POST};

    graph = postpre_sgraph{i};
    Cgraph = postpre_Cgraph{i};
    Mgraph = postpre_Mgraph{i};   
    
    W = zeros(Nc{i}{POST}, Nc{i}{PRE});
    WM = zeros(Nc{i}{POST}, Nc{i}{PRE});
    
    for pidx_f = 1:npools_pre  % from
        psize_f = Ns_pre(pidx_f);
        bs_f = border_starts_pre(pidx_f);
        be_f = border_ends_pre(pidx_f);
        
        for pidx_t = 1:npools_post  % to
            this_g = graph(pidx_t,pidx_f);
            this_c = Cgraph(pidx_t, pidx_f);
            
            psize_t = Ns_post(pidx_t);
            bs_t = border_starts_post(pidx_t);
            be_t = border_ends_post(pidx_t);
            A = zeros(psize_t, psize_f);
            
            rpidxs = [];
            if this_c > 0
                % Slow to go row by row, but also guarentees exactly this_c connections in each row.  I'm thinking of working with such small minipools
                % that I really don't want any fluctuation here.
                for tidx = 1:psize_t
                    rpidxs = randperm(psize_f);
                    rpidxs = sort(rpidxs(1:this_c));
                    A(tidx,rpidxs) = randn(1,this_c) * this_g / sqrt(this_c);
                end
                
                W(bs_t:be_t, bs_f:be_f) = A;
            end
            
            % The modification mask follows the connectivity structure, even if g is 0 for a given block.  This way one can initialize a zero, but sparse
            % set of modifiable connections.
            M_rr = Mgraph(pidx_t, pidx_f);
            modmat = zeros(psize_t,psize_f);
            if conn_struct.modMaskFollowsSparseness(i) && this_c > 0
                modmat(find(A)) = 1;
            else
                modmat(:) = 1;
            end
            WM(bs_t:be_t, bs_f:be_f) = M_rr * modmat;
        end
    end
    
    if conn_struct.doSparsify(i)
        W = sparse(W);
        WM = sparse(WM);
    end
    if i == I2R
        n_Wru_v = W;
        n_WruM_v = WM;
    elseif i == R2R
        n_Wrr_n = W;
        n_WrrM_n = WM;
    elseif i == R2Z
        m_Wzr_n = W;
        m_WzrM_n = WM;        
    end
end


if do_plot
    figure;
    subplot 231;
    imagesc(abs(n_Wru_v)); colormap jet; colorbar;
    title('n_Wru_v');
    
    subplot 234;
    imagesc(abs(n_WruM_v)); colormap jet; colorbar;
    title('n_WruM_v');
    
    subplot 232;
    imagesc(abs(n_Wrr_n)); colormap jet; colorbar;
    title('n_Wrr_n');
    
    subplot 235;
    imagesc(abs(n_WrrM_n)); colormap jet; colorbar;
    title('n_WrrM_n');
    
    subplot 233;
    imagesc(abs(m_Wzr_n)); colormap jet; colorbar;
    title('n_Wzr_n');
    
    subplot 236;
    imagesc(abs(m_WzrM_n)); colormap jet; colorbar;
    title('n_WzrM_n');
end
