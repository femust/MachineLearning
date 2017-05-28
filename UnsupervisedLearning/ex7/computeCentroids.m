function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);



% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% 
% for i=1:m
%     if idx(i)==1
%         centroids(1,1)=centroids(1,1)+X(i,1);
%         centroids(1,2)=centroids(1,2)+X(i,2);
%         count1=count1+1;
%     elseif idx(i)==2
%         centroids(2,1)=centroids(2,1)+X(i,1);
%         centroids(2,2)=centroids(2,2)+X(i,2);
%         count2=count2+1;
%     elseif idx(i)==3
%         centroids(3,1)=centroids(3,1)+X(i,1);
%         centroids(3,2)=centroids(3,2)+X(i,2);
%         count3=count3+1;
%     end
% end
% 
% centroids=[centroids(1,:)./count1; centroids(2,:)./count2;centroids(3,:)./count3]

for i=1:K
    count=sum(idx==i);
  %  if count==0
   %     centroids(i,1)=0;
    %    centroids(i,2)=0;
   % else
    centroids(i,1)=sum((idx==i).*X(:,1))./count;
    centroids(i,2)=sum((idx==i).*X(:,2))./count;
   % end
end

    
    





% =============================================================


end

