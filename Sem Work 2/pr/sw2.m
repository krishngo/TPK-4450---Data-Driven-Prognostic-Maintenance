%% function simuTraj
close all;
clear all;

%% problem 2
A = importdata('monitoring.txt');
Y = A(:,2);
t = A(:,1);
alpha = inv(transpose(t)*t)*transpose(t)*Y;
Yt = alpha*t;
figure(1);
hold on;
scatter( t,Y);
plot(t,Yt);
legend('observations','fitted line');

%% problem3
error=Y-Yt; 
mu = sum(error)/length(error);
sigma = std(error);

%% problem4
t2=linspace(60,160);
Yt2=alpha*transpose(t2);
traj = [];
figure(1);
hold on;
for i = 1:10000
    error2=normrnd(mu,sigma,100,1);
    Y2=Yt2+error2;
    plot(t2,Y2);
    traj=[traj;transpose(Y2)];
end

%%  problem5
pass_time = [];
for i = 1:length(traj)
    pass_index = traj(i,:) > 10;
    pass_vector = t2(pass_index);
    pass_time = [pass_time; min(pass_vector)];
    RUL = pass_time-60;
end

figure(2);
hold on;
histogram(RUL,30, 'Normalization', 'pdf');

%% problem 6
Inc = diff(Y);
mu_inc = sum(Inc)/length(Inc);
sigma_inc = std(Inc);
%% problem 7
t3=linspace(60,500,440);
yt_j=6.19;
traj2 = [];
figure(3);
hold on;
scatter( t,Y);
plot(t, Yt);
for i=1:10000
    random_inc = normrnd(mu_inc,sigma_inc,length(t3),1);
    inc_sum = cumsum(random_inc);
    Yt3 = yt_j + inc_sum;
    plot(t3,Yt3);
    traj2 = [traj2;transpose(inc_sum)];
end 
%% problem 8
pass_time2 = [];
for i = 1:length(traj2)
    pass_index2 = traj2(i,:) > 10;
    pass_vector2 = t3(pass_index2);
    pass_time2 = [pass_time2; min(pass_vector2)];
end
RUL2 = pass_time2-60;
figure(4);
hold on;
histogram(RUL2,60, 'Normalization', 'pdf');
%% problem 9
L = 10;
mu_i_hat = (L-yt_j)/mu_inc;
lambda_i_hat = ((L-yt_j)/sigma_inc)^2;
pdf_vec = [];

x_vec = linspace(0.1,500,10000);
for i= 1:length(x_vec);
    a= lambda_i_hat/(2*pi*x_vec(i)^3);
    b= lambda_i_hat/(2*mu_i_hat^2);
    c= (x_vec(i)-mu_i_hat)^2/x_vec(i);
    f_t=sqrt(a)*exp(-b*c);
    pdf_vec= [pdf_vec, f_t];
end
figure(4);
hold on;
plot(x_vec,pdf_vec);
%% problem 10;

t_j = 60;
t10=94;
% linear trend
count = sum(RUL<34);
prob = count/length(RUL);

%stochastic process
count2 = sum(RUL<34);
prob2 = count2/length(RUL2);

%inverse gaussian distribution
fun = @(h)(sqrt(lambda_i_hat./(2.*pi.*h.^3)).*exp((-lambda_i_hat.^2)).*(h-mu_i_hat.^2./h));
q=integral(fun,0,34);
