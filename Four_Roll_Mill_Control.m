DefOmega=[-1,1,1,-1]; % Default angular velocities
WiggleRoom=0.7*[1,1,1,1]; % WiggleRoom(i) is the semi-width of DefOmega(i)'s wiggle room 
dt=0.125* 0.20; % Time step
d1=0.125* 0.04; % Delay between picture acquisition and speed correction. We call this t_1 in the paper
d2=0.125* 0.04; % Time to reset the velocities. We call this t_2 in the paper
lag=0.125*0.10; % This is what we call t_lag in the paper
InitialState=[-0.03,0.02]; % Initial position

r=0.8; % Radius (same fol all cylinders)

c11=[-1,1];
c12=[1,1];
c21=[-1,-1];
c22=[1,-1]; % Position of centres (normalised)

u=@(x,Omega) r^2*(Omega(1)*[1-x(2),x(1)+1]/norm(x-c11)^2+Omega(2)*[1-x(2),x(1)-1]/norm(x-c12)^2+Omega(3)*[-1-x(2),x(1)+1]/norm(x-c21)^2+Omega(4)*[-1-x(2),x(1)-1]/norm(x-c22)^2);
% Velocity of the fluid

% See text at the top for recommended values of learning parameters
alphaD=10;
alphaT=10; % Step sizes for gradient ascent
gamma=0.95; % Discount factor
L=0.05; % Half-side length of big square.
p=1; % Peakedness of reward function. Positive.
ArraySize=4; % To approximate the policy and the value function

D=zeros(ArraySize);
T=zeros(ArraySize,ArraySize,ArraySize);

a=@(x) x.^(0:ArraySize-1)';
b=@(y) y.^(0:ArraySize-1)';
W=@(w) w.^(0:ArraySize-1)'; % Feature vectors

EpisodeLength=40; % Largest number of moves per episode
NumberOfEpisodes=100; 

for episode=1:NumberOfEpisodes
    episode
    State=InitialState; % Initial state
    I=1;
    
    for t=1:EpisodeLength
        q=mod(sign(State)*[1;-2],5); % This is the rotor to the right of the current one
        x=State(1);
        y=State(2);
       
        A=Action(a(x),b(y),T,L); % Take an action according to the current policy
 
        
        NextState=State;
        h=dt/20;
        time=0;
        
        % Use RK4 to determine the next state
        
        for n=0:20
            k1=u(NextState,DefOmega+(1:4==q)*hat(time,lag,d1,d2,dt,(1/L)*WiggleRoom(q)*A));
            k2=u(NextState+(h/2)*k1,DefOmega+(1:4==q)*hat(time+h/2,lag,d1,d2,dt,(1/L)*WiggleRoom(q)*A));
            k3=u(NextState+(h/2)*k2,DefOmega+(1:4==q)*hat(time+h/2,lag,d1,d2,dt,(1/L)*WiggleRoom(q)*A));
            k4=u(NextState+h*k3,DefOmega+(1:4==q)*hat(time+h,lag,d1,d2,dt,(1/L)*WiggleRoom(q)*A));
            NextState=NextState+(h/6)*(k1+2*k2+2*k3+k4);
            time=time+h;
        end
        
        x1=NextState(1);
        y1=NextState(2);
        
        if abs(x1)>L || abs(y1)>L % Discontinue if we land outside the big square
           break
        end
        
        Rew=exp(-p*(1+dot(NextState-State,State)/(norm(State)*norm(NextState-State))));
        
        delta=Rew+gamma*a(x1)'*D*b(y1)-a(x)'*D*b(y); % Update parameters via gradient ascent
        D=D+alphaD*I*delta*a(x)*b(y)';
        T=T+alphaT*I*delta*a(x)*b(y)'.*permute(W(A),[3,2,1]);
        
        I=gamma*I;
        State=NextState;
    end
end
%%
function w=Action(a,b,T,L) % a,b column vectors
T(1,1,1)=0;
k=(1-L)*abs(T(1,1,1))-L*abs(T(1,1,1))+L*sum(abs(T),'all');
alpha=0;
n=length(a);
while rand()>alpha
    w=L*(2*rand-1);
    alpha=exp(sum(T.*a.*permute(b,[2,1,3]).*permute(w.^(0:n-1)',[3,2,1]),'all')-k);
end
end
%% 
% This function computes the velocity of the cylinders when they are acted on.

function x=hat(s,lag,d1,d2,dt,C)
x=C;
if s<lag
    x=0;
end
if s>=lag && s< lag+d1
    x=(C/d1)*(s-lag);
end
if s>=dt-d2
    x=C/d2*(dt-d2-s)+C;
end
end
