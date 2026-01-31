"""
Hyperbolic Temporal RE-GCN Main Training Script.

This script provides the entry point for training and evaluating the
Hyperbolic Temporal RE-GCN model for temporal knowledge graph completion.

OPTIMIZATIONS (v2):
- Added comprehensive logging for debugging and analysis
- Added new command-line options for architecture improvements
- Added gradient statistics logging
- Added embedding statistics logging

Usage:
    python hyperbolic_main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 \\
        --lr 0.001 --n-layers 2 --n-hidden 200 --self-loop --layer-norm \\
        --entity-prediction --relation-prediction --gpu 0
"""

import argparse
import os
import sys
import time
import math
import random
import logging

import dgl
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append("..")

from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list
from hyperbolic_src.hyperbolic_model import HyperbolicRecurrentRGCN

# Set up logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set specific loggers
    logging.getLogger("hyperbolic_model").setLevel(log_level)
    logging.getLogger("hyperbolic_ops").setLevel(log_level)
    
    return logging.getLogger("hyperbolic_main")


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list, model_name, static_graph, mode, args, logger=None):
    """
    Test the model on given test data.
    
    Args:
        model: The model to test
        history_list: All input history snapshots
        test_list: Test triple snapshots
        num_rels: Number of relations
        num_nodes: Number of nodes
        use_cuda: Whether to use CUDA
        all_ans_list: Dict for entity filter MRR
        all_ans_r_list: Dict for relation filter MRR
        model_name: Model checkpoint path
        static_graph: Static graph
        mode: "train" or "test"
        args: Command line arguments
        
    Returns:
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    
    if mode == "test":
        # Load model checkpoint
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model: {}. Using best epoch: {}".format(model_name, checkpoint['epoch']))
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    # Get history for testing
    input_list = [snap for snap in history_list[-args.test_history_len:]]
    
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        # Build history graphs
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) 
                        for g in input_list]
        
        test_triples_input = torch.LongTensor(test_snap)
        if use_cuda:
            test_triples_input = test_triples_input.cuda().to(args.gpu)
        
        # Predict
        test_triples, final_score, final_r_score = model.predict(
            history_glist, num_rels, static_graph, test_triples_input, use_cuda
        )
        
        # Compute metrics
        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )
        
        # Store results
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)
        
        # Update history
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(
                    test_triples, num_nodes, num_rels, final_score, args.topk
                )
            else:
                predicted_snap = utils.construct_snap_r(
                    test_triples, num_nodes, num_rels, final_r_score, args.topk
                )
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    
    # Log final metrics
    if logger:
        logger.info(f"Test Results - MRR (raw): {mrr_raw:.4f}, MRR (filter): {mrr_filter:.4f}")
        logger.info(f"Test Results - Rel MRR (raw): {mrr_raw_r:.4f}, Rel MRR (filter): {mrr_filter_r:.4f}")
    
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def _compute_radius_targets(triple_snapshots, num_nodes, alpha=0.5, beta=0.5,
                            radius_min=0.5, radius_max=3.0):
    degrees = [set() for _ in range(num_nodes)]
    freq = np.zeros(num_nodes, dtype=np.float64)
    for snapshot in triple_snapshots:
        if len(snapshot) == 0:
            continue
        src = snapshot[:, 0]
        dst = snapshot[:, 2]
        freq += np.bincount(src, minlength=num_nodes)
        freq += np.bincount(dst, minlength=num_nodes)
        for s, d in zip(src, dst):
            degrees[s].add(d)
            degrees[d].add(s)
    degree_counts = np.array([len(neighbors) for neighbors in degrees], dtype=np.float64)
    abstract_score = alpha * np.log1p(degree_counts) + beta * np.log1p(freq)
    if abstract_score.max() - abstract_score.min() < 1e-9:
        normed = np.full_like(abstract_score, 0.5)
    else:
        normed = (abstract_score - abstract_score.min()) / (abstract_score.max() - abstract_score.min())
    return radius_min + (radius_max - radius_min) * normed


def run_experiment(args):
    """
    Run the training/testing experiment with comprehensive logging.
    
    Args:
        args: Command line arguments
        
    Returns:
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = f"hyperbolic_training_{args.dataset}.log" if args.log_file else None
    logger = setup_logging(log_level, log_file)
    
    logger.info("=" * 60)
    logger.info("Hyperbolic Temporal RE-GCN Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {args}")
    
    # Load data
    logger.info("Loading graph data...")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"  - Entities: {num_nodes}")
    logger.info(f"  - Relations: {num_rels}")
    logger.info(f"  - Train snapshots: {len(train_list)}")
    logger.info(f"  - Valid snapshots: {len(valid_list)}")
    logger.info(f"  - Test snapshots: {len(test_list)}")
    
    # Load answer lists for filtering
    all_ans_list_test = utils.load_all_answers_for_time_filter(
        data.test, num_rels, num_nodes, False
    )
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(
        data.test, num_rels, num_nodes, True
    )
    all_ans_list_valid = utils.load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, False
    )
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, True
    )
    
    # Model name for checkpointing (updated to include new parameters)
    use_residual = not args.disable_residual
    model_name = "hyperbolic-{}-{}-{}-ly{}-c{}-his{}-weight:{}-angle:{}-dp{}|{}|{}|{}-res{}-lc{}-cmax{}-cw{}-gpu{}".format(
        args.dataset, args.encoder, args.decoder, args.n_layers,
        args.curvature, args.train_history_len, args.weight, args.angle,
        args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout,
        int(use_residual), int(args.learn_curvature), args.curvature_max, args.curvature_warmup_epochs,
        args.gpu
    )
    model_state_file = '../models/' + model_name
    logger.info(f"Model checkpoint: {model_state_file}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    
    # Handle static graph
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            "../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False
        ))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        if use_cuda:
            static_node_id = static_node_id.cuda(args.gpu)
        logger.info(f"Static graph loaded: {num_static_rels} relations, {num_words} words")
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    
    # Create model with new parameters
    logger.info("Creating Hyperbolic Recurrent RGCN model...")
    radius_targets = _compute_radius_targets(
        train_list, num_nodes, alpha=args.radius_alpha, beta=args.radius_beta,
        radius_min=args.radius_min, radius_max=args.radius_max
    )
    logger.info(
        f"Radius targets: min={radius_targets.min():.4f}, "
        f"max={radius_targets.max():.4f}, mean={radius_targets.mean():.4f}"
    )
    model = HyperbolicRecurrentRGCN(
        decoder_name=args.decoder,
        encoder_name=args.encoder,
        num_ents=num_nodes,
        num_rels=num_rels,
        num_static_rels=num_static_rels,
        num_words=num_words,
        h_dim=args.n_hidden,
        opn=args.opn,
        sequence_len=args.train_history_len,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        c=args.curvature,
        self_loop=args.self_loop,
        skip_connect=args.skip_connect,
        layer_norm=args.layer_norm,
        input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout,
        weight=args.weight,
        discount=args.discount,
        angle=args.angle,
        use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda,
        gpu=args.gpu,
        analysis=args.run_analysis,
        # New optimization parameters
        learn_curvature=args.learn_curvature,
        use_residual_evolution=use_residual,
        radius_target=radius_targets,
        radius_lambda=args.radius_lambda,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        radius_epsilon=args.radius_epsilon,
        curvature_min=args.curvature_min,
        curvature_max=args.curvature_max
    )
    
    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        logger.info(f"Model moved to GPU {args.gpu}")
    
    # Build static graph
    if args.add_static_graph:
        static_graph = build_sub_graph(
            len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu
        )
    
    # Optimizer - use same setup as original RE-GCN for consistency
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    logger.info(f"Optimizer: Adam, lr={args.lr}, weight_decay=1e-5")
    
    # Testing mode
    if args.test and os.path.exists(model_state_file):
        logger.info("Starting evaluation mode...")
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(
            model, train_list + valid_list, test_list, num_rels, num_nodes,
            use_cuda, all_ans_list_test, all_ans_list_r_test, model_state_file,
            static_graph, "test", args, logger
        )
    elif args.test and not os.path.exists(model_state_file):
        logger.warning("Model not found, switching to training mode")
        args.test = False
    
    # Training mode
    if not args.test:
        logger.info("=" * 40)
        logger.info("Starting Training")
        logger.info("=" * 40)
        best_mrr = 0
        best_epoch = 0
        early_stop_patience = 20
        training_start_time = time.time()
        
        for epoch in range(args.n_epochs):
            epoch_start_time = time.time()
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []
            losses_radius = []
            
            # Shuffle training order
            idx = list(range(len(train_list)))
            random.shuffle(idx)
            
            if args.learn_curvature:
                if args.curvature > args.curvature_max:
                    logger.warning(
                        "Initial curvature is greater than curvature_max; "
                        "adjusting warmup start to curvature_max."
                    )
                    args.curvature = args.curvature_max
                if args.curvature_warmup_epochs > 0 and epoch < args.curvature_warmup_epochs:
                    warmup_progress = (epoch + 1) / args.curvature_warmup_epochs
                    current_max = args.curvature + (args.curvature_max - args.curvature) * warmup_progress
                    model.set_curvature_bounds(curvature_max=current_max)
                else:
                    model.set_curvature_bounds(curvature_max=args.curvature_max)

            for train_sample_num in tqdm(idx, desc=f"Epoch {epoch}"):
                if train_sample_num == 0:
                    continue
                
                output = train_list[train_sample_num:train_sample_num + 1]
                
                # Get history
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0:train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:train_sample_num]
                
                # Build history graphs
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                                for snap in input_list]
                
                # Prepare output
                output = [torch.from_numpy(_).long() for _ in output]
                if use_cuda:
                    output = [o.cuda() for o in output]
                
                # Compute loss
                loss_e, loss_r, loss_static, loss_radius = model.get_loss(
                    history_glist, output[0], static_graph, use_cuda
                )
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static + loss_radius
                
                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())
                losses_radius.append(loss_radius.item())
                if args.run_analysis:
                    logger.debug(f"Radius loss: {loss_radius.item():.4f}")
                
                # Backprop
                loss.backward()
                
                # Log gradient statistics if in analysis mode
                if args.run_analysis and train_sample_num % 100 == 0:
                    model.log_gradient_stats()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch summary
            epoch_summary = (f"Epoch {epoch:04d} | "
                           f"Loss: {np.mean(losses):.4f} | "
                           f"E/R/S/Rad: {np.mean(losses_e):.4f}/{np.mean(losses_r):.4f}/"
                           f"{np.mean(losses_static):.4f}/{np.mean(losses_radius):.4f} | "
                           f"Best MRR: {best_mrr:.4f} | "
                           f"Time: {epoch_time:.1f}s")
            logger.info(epoch_summary)
            print(epoch_summary)
            
            # Log model-specific training summary
            if args.run_analysis:
                training_summary = model.get_training_summary()
                logger.info(f"Training summary: {training_summary}")
            
            # Validation
            if epoch and epoch % args.evaluate_every == 0:
                logger.info(f"Running validation at epoch {epoch}...")
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(
                    model, train_list, valid_list, num_rels, num_nodes,
                    use_cuda, all_ans_list_valid, all_ans_list_r_valid,
                    model_state_file, static_graph, mode="train", args=args, logger=logger
                )
                
                # Log validation metrics
                logger.info(f"Validation - MRR: raw={mrr_raw:.4f}, filter={mrr_filter:.4f}")
                logger.info(f"Validation - Rel MRR: raw={mrr_raw_r:.4f}, filter={mrr_filter_r:.4f}")
                
                current_mrr = mrr_raw_r if args.relation_evaluation else mrr_raw
                if current_mrr > best_mrr:
                    best_mrr = current_mrr
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                              model_state_file)
                    logger.info(f"New best model saved! MRR: {best_mrr:.4f}")
                elif epoch - best_epoch >= early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch}: no improvement in {early_stop_patience} epochs.")
                    break
        
        # Log training time
        total_training_time = time.time() - training_start_time
        logger.info(f"Training completed in {total_training_time/60:.1f} minutes")
        
        # Final test
        logger.info("=" * 40)
        logger.info("Final Evaluation on Test Set")
        logger.info("=" * 40)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(
            model, train_list + valid_list, test_list, num_rels, num_nodes,
            use_cuda, all_ans_list_test, all_ans_list_r_test, model_state_file,
            static_graph, mode="test", args=args, logger=logger
        )
        
        # Log final results
        logger.info("=" * 40)
        logger.info("Final Test Results")
        logger.info("=" * 40)
        logger.info(f"Entity Prediction - MRR: raw={mrr_raw:.4f}, filter={mrr_filter:.4f}")
        logger.info(f"Relation Prediction - MRR: raw={mrr_raw_r:.4f}, filter={mrr_filter_r:.4f}")
        logger.info(f"Best epoch: {best_epoch}, Best MRR: {best_mrr:.4f}")
    
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperbolic Temporal RE-GCN')
    
    # Basic settings
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--test", action='store_true', default=False, help="Test mode")
    parser.add_argument("--run-analysis", action='store_true', default=False, help="Run analysis")
    parser.add_argument("--multi-step", action='store_true', default=False, help="Multi-step inference")
    parser.add_argument("--topk", type=int, default=10, help="Top-k for multi-step")
    parser.add_argument("--add-static-graph", action='store_true', default=False, help="Use static graph")
    parser.add_argument("--relation-evaluation", action='store_true', default=False, help="Evaluate by relation")
    
    # Hyperbolic space settings
    parser.add_argument("--curvature", type=float, default=0.01, help="Curvature of hyperbolic space")
    parser.add_argument("--learn-curvature", action='store_true', default=False, help="Learn curvature during training (NEW)")
    parser.add_argument("--curvature-min", type=float, default=1e-4, help="Minimum curvature for scheduling")
    parser.add_argument("--curvature-max", type=float, default=1e-1, help="Maximum curvature for scheduling")
    parser.add_argument("--curvature-warmup-epochs", type=int, default=0, help="Warmup epochs for curvature scheduling")
    parser.add_argument("--disable-residual", action='store_true', default=False, help="Disable residual temporal radius evolution")
    parser.add_argument("--radius-alpha", type=float, default=0.5, help="Weight for degree-based radius target")
    parser.add_argument("--radius-beta", type=float, default=0.5, help="Weight for frequency-based radius target")
    parser.add_argument("--radius-min", type=float, default=0.5, help="Minimum static radius")
    parser.add_argument("--radius-max", type=float, default=3.0, help="Maximum static radius")
    parser.add_argument("--radius-lambda", type=float, default=0.02, help="Radius supervision loss weight")
    parser.add_argument("--radius-epsilon", type=float, default=0.1, help="Max temporal radius perturbation")
    
    # Encoder settings
    parser.add_argument("--weight", type=float, default=1, help="Weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7, help="Weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1, help="Discount of static constraint weight")
    parser.add_argument("--angle", type=int, default=10, help="Evolution speed angle")
    parser.add_argument("--encoder", type=str, default="hyperbolic_uvrgcn", help="Encoder method")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False, help="Use skip connections")
    parser.add_argument("--n-hidden", type=int, default=200, help="Hidden dimension")
    parser.add_argument("--opn", type=str, default="sub", help="Operation for CompGCN")
    parser.add_argument("--n-bases", type=int, default=100, help="Number of weight bases")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--self-loop", action='store_true', default=True, help="Use self-loop")
    parser.add_argument("--layer-norm", action='store_true', default=False, help="Use layer normalization")
    parser.add_argument("--relation-prediction", action='store_true', default=False, help="Enable relation prediction")
    parser.add_argument("--entity-prediction", action='store_true', default=False, help="Enable entity prediction")
    
    # Training settings
    parser.add_argument("--n-epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--evaluate-every", type=int, default=1, help="Evaluation frequency")
    
    # Decoder settings
    parser.add_argument("--decoder", type=str, default="hyperbolic_convtranse", help="Decoder method")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="Input dropout")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="Hidden dropout")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="Feature dropout")
    
    # Sequence settings
    parser.add_argument("--train-history-len", type=int, default=10, help="Training history length")
    parser.add_argument("--test-history-len", type=int, default=20, help="Testing history length")
    
    # Logging settings (NEW)
    parser.add_argument("--verbose", action='store_true', default=False, help="Enable verbose/debug logging")
    parser.add_argument("--log-file", action='store_true', default=False, help="Save logs to file")
    
    args = parser.parse_args()
    print(args)
    
    run_experiment(args)
