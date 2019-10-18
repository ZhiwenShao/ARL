clc,clear;

model_path = './';
data_name = 'BP4D';

test_label = importdata(['../', data_name, '_intensity_AUoccur.txt']);

start_iter = 1;
n_iters = 12;
missing_label = 9;

fid = fopen([model_path, data_name, '_res_all_', num2str(start_iter), '.txt'], 'w');

for _iter=start_iter:n_iters

	pred_label = importdata([model_path, data_name, '_test_AU_intensity_pred-', num2str(_iter), '_all_.txt']);
	pred_label = round(pred_label);
	for i=1:size(test_label,2)
	
	    curr_pred_label = pred_label(:,i)';
	    curr_test_label = test_label(:,i)';
	    
	    select_ind =find(curr_test_label~=missing_label);
	    curr_pred_label=curr_pred_label(select_ind);
	    curr_test_label=curr_test_label(select_ind);
	    
	    % compute evaluation metrics
	    ee = curr_pred_label - curr_test_label; 
	    dat = [curr_pred_label; curr_test_label]'; 
	    abs_test(1,i) = sum(abs(ee))/length(ee); % Mean Absolute Error (MAE)
	    icc_test(1,i) = ICC(3,'single',dat); % Intra-Class Correlation (ICC)
	    
	end
	
	fprintf(fid, '%d\t%.2f\t%.2f\n', _iter, mean(icc_test), mean(abs_test));
end

fclose(fid);