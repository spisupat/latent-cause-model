%% Latent cause model of associative learning
% Uses a chinese restaurant prior over possible latent causes, and
% particle filtering to approximate latent cause inference, based on
% Gershman, Blei & Niv 2010
%
% Author: Sashank Pisupati
% Version: 2.1.0


%% Model parameters
clear all

% Task specification
task = 'deterministic_reward';
% task = 'probabilistic_reward';

% Block specification: Comma separated characters indicating feature presence, alphabetically last character denotes outcome.
blocks = {'A+','A'};               % E.g.: A+ denotes presence of A and reward(+), AX denotes presence of A,X, absence of reward.
% probs = [0.8,0,1;0.8,0.5];          % For probabilistic_reward only, specify probabilities of each character
nTrialsPerBlock = 30;

% CRP/Particle filter parameter
scheme = 'importanceResampling';
alpha = 0.1;                        % CRP concentration parameter
aPrior = 1;                         % Prior pseudocounts of feature presence
bPrior = 1;                         % Prior pseudocounts of feature absence
nVisCauses = 10;                    % Number of latent causes to visualize
nParticles = 1000;                  % Number of particles

%% Stimulus design

% Design trial-wise feature array (stimuli+outcomes) using specified block structure
switch task
    case 'deterministic_reward'
        %Default: conditioned inhibition (A+/AX-)
        if ~exist('blocks','var')
            blocks = {'A+','AX'};
        end
        features = unique([blocks{:}]);
        features = [features(2:end),features(1)];
        lFeatures = num2cell(features);
        nFeatures = length(lFeatures);          % Number of features
        f = zeros(nFeatures,nTrialsPerBlock);   % Active features
        for b = 1:length(blocks)
            for c = 1:length(blocks{b})
                f(strcmp(num2cell(blocks{b}(c)),lFeatures),(b-1)*nTrialsPerBlock+1:b*nTrialsPerBlock)=1;
            end
        end
        
    case 'probabilistic_reward'
        % Default: simple conditioning (A+ @ 80%)
        if ~exist('blocks','var')
            blocks = {'A+'};
        end
        features = unique([blocks{:}]);
        if ~exist('probs','var')
            probs = 0.8*ones(length(blocks),length(features));
        end
        features = [features(2:end),features(1)];
        lFeatures = num2cell(features);
        nFeatures = length(lFeatures);          % Number of features
        f = zeros(nFeatures,nTrialsPerBlock);   % Active features
        for b = 1:length(blocks)
            for c = 1:length(blocks{b})
                for n = 1:nTrialsPerBlock
                    f(strcmp(num2cell(blocks{b}(c)),lFeatures),(b-1)*nTrialsPerBlock+n)=rand<probs(b,c);
                end
            end
        end
end
nTrials = size(f,2);

% Plot stimulus structure
figure(1);
set(gcf,'color','w');
subplot(3,2,1)
imagesc(f)
colormap(bone)
caxis([0,1])
xlabel('trials')
set(gca,'ytick',[1:nFeatures])
set(gca,'yticklabels',lFeatures)
title('Observations')
set(gca,'fontSize',18)


%% Initialization

% Initialize latent cause assignments, counts(t=0)
nMaxCauses = nTrials;                   % Maximum number of possible latent causes
particles = zeros(nParticles,nTrials);  % Latent cause assignments i.e. particles
nC = zeros(nParticles,nMaxCauses);      % Latent cause counts for each particle

% Initialize sufficient statistics (counts) of observation probabilities with beta prior
nFC = zeros(nFeatures,nMaxCauses,nParticles)+aPrior;    % Feature|cause presence pseudocounts
bFC = zeros(nFeatures,nMaxCauses,nParticles)+bPrior;    % Feature|cause absence pseudocounts

% Plot prior over observation probabilities
subplot(3,2,2)
plot(betapdf(0:0.01:1,aPrior,bPrior))
title('Priors')
set(gca,'fontSize',18)

% Initialize likelihood & weights
lik = zeros(nParticles,nMaxCauses);     % Likelihood
wt = ones(nParticles,1)/nParticles;     % Importance weights
rt = ones(nParticles,1)/nParticles;     % Outcome predictive weights
rProb = zeros(nParticles,1);            % Outcome probablities

% Initialize estimates averaged across particles
cEst = zeros(nMaxCauses,nTrials);       % Estimated posterior on causes
rEst = zeros(1,nTrials);                % Estimated outcome probability
phiEst = zeros(nFeatures,nTrials);      % Estimated observation probabilities


%% Inference

%Loop over trials
for t = 1:nTrials
    obs = logical(f(:,t)); %Observations on current trial
    
    switch scheme
        
        case 'importanceResampling'
            % Sequential Importance Resampling: Sample prior set of particles before each trial,
            % Weigh by importance (likelihood) to estimate posterior, importance-weighted resample
            
            % Sample particles(t), counts(t) from CRP prior, based on counts(t-1)
            [particles(:,t),nC] = generateCRprior(nC,alpha);
            
            %Loop over particles
            for l = 1:nParticles
                %Current hypothesized latent cause
                k = particles(l,t);
                
                % Value prediction (before seeing outcome):
                % Likelihood of non-outcome features given hypothesized latent cause i.e. Predictive weights
                rt(l) = prod((obs(1:end-1).*nFC(1:end-1,k,l) + ~obs(1:end-1).*bFC(1:end-1,k,l))./(nFC(1:end-1,k,l)+bFC(1:end-1,k,l)),1);
                % Probability of outcome feature given hypothesized latent cause
                rProb(l) = nFC(end,k,l)./(nFC(end,k,l)+bFC(end,k,l));
                
                % Likelihood of *all* observed features given hypothesized latent cause i.e. Importance weights
                wt(l) = prod((obs.*nFC(:,k,l) + ~obs.*bFC(:,k,l))./(nFC(:,k,l)+bFC(:,k,l)),1);
                
                % Update observation probabilities
                nFC(:,k,l) = nFC(:,k,l) + double(obs);
                bFC(:,k,l) = bFC(:,k,l) + double(~obs);
            end
            % Normalize weights
            wt = wt/sum(wt);
            rt = rt/sum(rt);
            
            % Predictive estimate of value i.e. outcome prob (before seeing outcome)
            rEst(t) = rt'*rProb;
            
            % Importance weighted estimate of posterior over causes
            cEst(:,t) = weightedHist(particles(:,t),wt,0:nMaxCauses);
            
            % Importance weighted estimate of posterior over observation probabilities
            for i = 1:nFeatures
                phiEst(i,:) = wt'*squeeze([(nFC(i,:,:)./(nFC(i,:,:)+bFC(i,:,:)))])';
            end
            
            
            % Importance weighted resampling of particles
            indices = randsample([1:nParticles]',nParticles,true,wt);
            particles = particles(indices,:);
            nC = nC(indices,:);
            nFC = nFC(:,:,indices);
            bFC = bFC(:,:,indices);
    end
    
    % Plot heatmap of estimated posterior over causes
    subplot(3,2,3)
    imagesc(cEst);caxis([0,1])
    ylim([0.5,nVisCauses])
    set(gca,'ytick',[1,round(nVisCauses/2),nVisCauses])
    title('p(Cause)')
    xlabel('trials')
    set(gca,'fontSize',18)
    
    % Plot graph of estimated posterior over causes
    subplot(3,2,5)
    plot(cEst(:,1:t)','LineWidth',1);
    set(gca, 'ColorOrder', hot(nVisCauses), 'NextPlot', 'replacechildren');
    ylim([0,1])
    xlim([1,nTrials])
    title('p(Cause)')
    xlabel('trials')
    set(gca,'fontSize',18)
    
    % Plot estimated value i.e. predicted outcome probability
    subplot(3,2,6)
    plot(rEst(1:t),'k-','LineWidth',1);
    hold on
    for b = 1:length(blocks)
        plot([b*nTrialsPerBlock,b*nTrialsPerBlock],[0,1],'k--');
    end
    hold off
    xlim([1,nTrials]);
    ylim([0,1]);
    set(gca,'fontSize',18)
    xlabel('trials')
    title('p(Outcome)')
    
    % Plot estimated observation probabilities
    set(gca,'fontSize',18)
    subplot(3,2,4)
    imagesc(phiEst);caxis([0,1])
    xlim([0.5,nVisCauses])
    title('p(Feature|Cause)')
    xlabel('Causes')
    set(gca,'ytick',[1:nFeatures])
    set(gca,'yticklabels',lFeatures)
    set(gca,'fontSize',18)
    drawnow
end

%% CHINESE RESTAURANT PROCESS

% Partition/Assignments generator - generates "M" assignments for 
% current trial (t) based on CRP prior, i.e. based on counts from trial (t-1),alpha
function [assignmentsT, nKT]= generateCRprior(nKTminus1,alpha)
% Partition/Assignment probability - requires m x K vector of counts "nk"
%   k = 1:size(nk,2);       %Index of causes
%   K = sum(nk>0,2);        %Number of assigned causes so far
%   N (=T-1) = sum(nk,2);   %Number of observations so far
%   pAssignCR = @(k,nk,K,alpha) (nk.*(k<=K)+alpha.*(k==K+1)+0.*(k>K+1))./(sum(nk,2)+alpha);
pAssignCR = @(nk,alpha) (nk.*(1:size(nk,2)<=sum(nk>0,2))+alpha.*(1:size(nk,2)==sum(nk>0,2)+1)+0.*(1:size(nk,2)>sum(nk>0,2)+1))./(sum(nk,2)+alpha);
thisProb = pAssignCR(nKTminus1,alpha);
thisCount = mnrnd(1,thisProb);
[row,col] = find(thisCount);
assignmentsT(row,1) = col;
nKT = nKTminus1 + thisCount;
end


%% Weighted histogram
function mass = weightedHist(x, weights, bins)
mass = zeros(length(bins)-1,1);
for j = 1:length(mass)
    inds = find(x>bins(j) & x<=bins(j+1));
    if ~isempty(inds)
        mass(j) = sum(weights(inds));
    end
end
end
