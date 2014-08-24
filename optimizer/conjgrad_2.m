function [xs, is, phis_to_go, pAp, simdata] = conjgrad_2( Afunc, b, x0, maxiters, miniters, Mdiag, simdata, varargin )

% This conjugate gradient routine was written by James Martens as part of his demo / release code for the Deep Learning ICML paper.  
% David Sussillo adapted it ever so slightly to fit his own optimizer.

epsilon = 5e-4;				% based on HF in DBN papers. it's the relative phi tolerance.

gapratio = 0.1;
mingap = 10;
inext_last = 1;
inext = 5;				% seems like 3 would be a better choice here, right? -DCS:2011/08/10
gamma = 1.3;
maxtestgap = max(ceil(maxiters * gapratio), mingap) + 1;
display_level = Inf;


optargin = size(varargin,2);
for i = 1:2:optargin			% perhaps a params structure might be appropriate here.  Haha.
    switch varargin{i}
        case 'gamma'
            gamma = varargin{i+1};
        case 'epsilon'
            epsilon = varargin{i+1};
        case 'displaylevel'
            display_level = varargin{i+1};
    end
end

%epsilon

phis = zeros(maxtestgap,1);

is = [];
xs = {};
phis_to_go = [];

%r = Afunc(x0) - b;
package = Afunc(x0, simdata);
Ap = package{1};
r = Ap - b;
simdata = package{2};

if ~isempty(Mdiag)
    y = r./Mdiag;
else
    y = r;
end

p = -y;
x = x0;

%phi is the value of the quadratic model
phi = 0.5*double((-b+r)'*x);		% 0.5 multiplies everything because r starts with a b.
%disp( ['iter ' num2str(0) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', phi = ' num2str( phi ) ]);

pAp = 0.0;
for i = 1:maxiters
    
    %compute the matrix-vector product.  This is where 95% of the work in
    %HF lies:
    package = Afunc(p, simdata);
    Ap = package{1};
    simdata = package{end};
    
    pAp = p'*Ap;
    %disp(num2str(pAp));
    %the Gauss-Newton matrix should never have negative curvature.  The Hessian easily could unless your objective is
    %convex
    if pAp <= 0
        disp(['Non-positive Curvature!: pAp = ', num2str(pAp,16)]);
        disp('Bailing...');
        break;
    end
    
    alpha = (r'*y)/pAp;
    
    x = x + alpha*p;
    r_new = r + alpha*Ap;
    
    if ~isempty(Mdiag)
        y_new = r_new./Mdiag;
    else
        y_new = r_new;
    end
    
    beta = (r_new'*y_new)/(r'*y);
    
    p = -y_new + beta*p;
    
    r = r_new;
    y = y_new;
    
    
    phi = 0.5*double((-b+r)'*x);
    phis( mod(i-1, maxtestgap)+1 ) = phi;
    
    %disp( ['iter ' num2str(i) ': ||x|| = ' num2str(double(norm(x))) ', ||r|| = ' num2str(double(norm(r))) ', ||p|| = ' num2str(double(norm(p))) ', phi = ' num2str( phi ) ]);
    
    testgap = max(ceil( i * gapratio ), mingap); % k in paper
    prevphi = phis( mod(i-testgap-1, maxtestgap)+1 ); %testgap steps ago
    
    if i == ceil(inext) || i == 5 || i == 3 || i == 1 % seems stupid to go from 5 to 1, but that's what gamma = 1.3 does. -DCS:2011/08/19
        is(end+1) = i; %#ok<AGROW>
        xs{end+1} = x; %#ok<AGROW>
        phis_to_go(end+1) = phi; %#ok<AGROW>
        inext_last = inext;
        inext = inext*gamma;
    end
    
    %the stopping criterion here becomes largely unimportant once you optimize your function past a certain point, as it
    %will almost never kick in before you reach i = maxiters.  And if the value of maxiters is set so high that this
    %never occurs, you probably have set it too high
    if i > testgap && prevphi < 0 && (phi - prevphi)/phi < epsilon*testgap && i >= miniters
        if display_level > 1
            disp(['CG breaking due to relative tolerance condition based on phi at iter: ', num2str(i) '.']);
        end
        break;
    end
    
    
end

if i ~= ceil(inext_last)
    is(end+1) = i;
    xs{end+1} = x;
    phis_to_go(end+1) = phi;
end
