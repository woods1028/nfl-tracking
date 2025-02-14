import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snowflake.snowpark.functions import col
import datetime

#%%

def prepare_clip_tensor(chunk_data, clip_id, model_features):

    clip_df = chunk_data[chunk_data['CLIP_ID'] == clip_id]

    clip_frames = int(len(clip_df)/(11*12))

    return_tensor = torch.tensor(
        (clip_df
         [model_features]
         .transpose()
         .to_numpy()
         .flatten()
         .reshape(len(model_features), clip_frames, 11, 12)
        ),
        dtype = torch.float32
    )

    return return_tensor

#%%

def get_chunk_dataset(chunk_data, chunk, coverage_mapping, model_features):
    """Create a dataset for a chunk of clips."""

    coverage_labels = (chunk_data
     .merge(
         coverage_mapping,
         on = "PFF_PASSCOVERAGE"
     )
     [['CLIP_ID','coverage']]
     .drop_duplicates()['coverage']
     .values
    )
    
    dataset = []

    for clip_id, label in zip(chunk, coverage_labels):

        #print(clip_id)

        clip_id_tensor = torch.tensor(clip_id,dtype = torch.long)

        clip_tensor = prepare_clip_tensor(chunk_data, clip_id, model_features)

        label_tensor = torch.tensor(label,dtype = torch.long)

        dataset.append((
            clip_tensor,
            label_tensor,
            clip_id_tensor
        ))
    
    return dataset

#%%

def process_batch(model, data, labels, criterion, optimizer=None, training=True):
    """Process a single batch of data."""
    if training:
        optimizer.zero_grad(set_to_none=True) 
    
    outputs = model(data)
    #probs = torch.softmax(outputs, dim = 1)
    loss = criterion(outputs, labels)
    
    if training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
    
    return loss.item(), correct, len(labels), predicted.tolist(), labels.tolist()

#%%

def custom_collate(batch):
    """
    Custom collate function that handles different sequence lengths.
    Each item in batch is (clip_tensor, label_tensor)
    """
    # Sort batch by sequence length (descending) to optimize for packed sequence
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    
    # Separate clips and labels
    clip_data, labels, clip_ids = zip(*batch)
    
    # Convert labels to tensor
    labels = torch.stack(labels)
    
    # Return as is - no padding needed
    return clip_data, labels, clip_ids

def process_chunk(model, chunk_data, chunk, coverage_mapping, model_features, 
                  criterion, batch_size, optimizer=None, training=True):

    dataset = get_chunk_dataset(chunk_data, chunk, coverage_mapping, model_features)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = custom_collate)
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, labels, _) in enumerate(loader):

        print(f'Batch ID: {batch_idx}')

        loss, correct, samples, preds, true_labels = process_batch(
            model, data, labels, criterion, optimizer, training
        )
        total_loss += loss
        total_correct += correct
        total_samples += samples
        all_preds.extend(preds)
        all_labels.extend(true_labels)
    
    return total_loss, total_correct, total_samples, all_preds, all_labels

#%%

def train_model_in_chunks(model_df,
                          model_features, coverage_mapping, 
                          ana_chunk_ids, assess_chunk_ids,
                          ana_clips, assess_clips,
                          coverages_to_model, class_counts,
                          num_epochs=10, batch_size=32,
                          ):
    
    # Initialize model and training components
    model = VideoClassifier(num_classes = len(coverages_to_model))

    class_weights = compute_class_weights(class_counts)
    
    criterion = FocalLoss(weight=class_weights, gamma=2.0, alpha = 0.25)

    # Initialize optimizer with momentum
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(ana_chunk_ids),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 
        'lr': []
    }
    
    for epoch in range(num_epochs):

        start_time = datetime.datetime.now()

        model.train()
        train_metrics = evaluate_chunks(
            model, model_df, ana_chunk_ids, 
            coverage_mapping, model_features, 
            ana_clips, criterion, batch_size, 
            scheduler,
            optimizer=optimizer, training=True
        )
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_chunks(
                model, model_df, assess_chunk_ids, 
                coverage_mapping, model_features, 
                assess_clips, criterion, batch_size * 2, 
                scheduler=None,
                optimizer=None, training=False
            )
        
        # Update and print metrics
        current_lr = optimizer.param_groups[0]['lr']
        update_history(history, train_metrics, val_metrics, current_lr)

        end_time = datetime.datetime.now()

        run_time = end_time - start_time

        results_df = pd.concat([
            pd.DataFrame({'actual':train_metrics['labels'],'pred':train_metrics['predictions']}).assign(set = 'train'),
            pd.DataFrame({'actual':val_metrics['labels'],'pred':val_metrics['predictions']}).assign(set = 'val')
        ])
        
        # Print epoch results
        print_epoch_results(epoch, run_time, train_metrics, val_metrics, current_lr)
    
    return model, history, results_df

#%%

def evaluate_chunks(
    model, model_df, chunk_ids, 
    coverage_mapping, model_features, 
    clips,criterion, batch_size, 
    scheduler=None,
    optimizer=None, training=False,
):
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for chunk_id in chunk_ids:

        print(f'Chunk: {chunk_id}')

        chunk = [int(x[0]) for x in clips if x[1] == chunk_id]
        chunk_df = model_df.filter(col('clip_id').in_(chunk))
        chunk_data = chunk_df.to_pandas()
        
        loss, correct, samples, preds, labels = process_chunk(
            model, chunk_data, chunk, coverage_mapping, model_features,
            criterion, batch_size, optimizer, training
        )
        
        total_loss += loss
        total_correct += correct
        total_samples += samples
        all_preds.extend(preds)
        all_labels.extend(labels)

        if training and scheduler is not None:
                scheduler.step()
    
    return {
        'loss': total_loss / len(chunk_ids),
        'accuracy': 100 * total_correct / total_samples,
        'predictions': all_preds,
        'labels': all_labels
    }

#%%

def compute_class_weights(class_counts):
    # Using inverse frequency with temperature scaling
    temp = 2.0
    weights = 1.0 / (class_counts + 1e-6)  # Add small epsilon to prevent division by 0
    weights = weights ** (1.0 / temp)  # Temperature scaling to smooth the weights
    class_weights =  weights / weights.sum()
    return torch.tensor(class_weights,dtype = torch.float32)

#%%

def update_history(history, train_metrics, val_metrics, lr):
    """Update training history dictionary."""
    history['train_loss'].append(train_metrics['loss'])
    history['train_acc'].append(train_metrics['accuracy'])
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['lr'].append(lr)

def print_epoch_results(epoch, run_time, train_metrics, val_metrics, lr):
    """Print results for an epoch."""
    print(f'Epoch: {epoch}')
    print(f'Runtime: {run_time}')
    print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
    print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
    print(f'Learning Rate: {lr}')
    print('-' * 50)

#%%

def setup_cuda_training():
    """Configure CUDA settings for optimal performance."""
    if not torch.cuda.is_available():
        return False
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.85)  # Reserve some VRAM for system
    torch.cuda.empty_cache()
    
    return True

#%%

def optimize_batch_processing(model, data, labels, criterion, optimizer=None, training=True):
    """Memory-efficient batch processing."""
    if training:
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    
    # Move data to GPU in chunks if needed
    if isinstance(data, list):
        data = [d.cuda() if len(d) < 1000 else d.cuda(non_blocking=True) for d in data]
    labels = labels.cuda()
    
    outputs = model(data)
    loss = criterion(outputs, labels)
    
    if training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        
    # Clear unnecessary tensors
    del outputs
    torch.cuda.empty_cache()
    
    return loss.item(), correct, len(labels), predicted.cpu().tolist(), labels.cpu().tolist()

def get_optimal_batch_size():
    """Calculate optimal batch size based on available VRAM."""
    vram = torch.cuda.get_device_properties(0).total_memory
    # For RTX 2060 6GB, aim for ~4GB VRAM usage
    # Approximately 80 samples with your model architecture
    return 80