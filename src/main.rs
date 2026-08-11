use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use ndarray::{ArrayView1, ArrayView2, s};

mod checkpoint;
mod snn;
mod xoshiro256pp;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
enum NormalizationArg {
    None,
    L1,
    L2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
enum DatasetArg {
    Mnist,
    Fashion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
enum MarkSplitArg {
    Validation,
    Test,
}

impl From<NormalizationArg> for snn::network::FeedforwardNormalization {
    fn from(value: NormalizationArg) -> Self {
        match value {
            NormalizationArg::None => snn::network::FeedforwardNormalization::None,
            NormalizationArg::L1 => snn::network::FeedforwardNormalization::L1,
            NormalizationArg::L2 => snn::network::FeedforwardNormalization::L2,
        }
    }
}

const PROGRESS_INTERVAL: usize = 100;
const MAX_TRAIN_AND_MARK_SAMPLES: usize = 60_000;
const MAX_TEST_SAMPLES: usize = 10_000;

#[derive(clap::Parser)]
struct Args {
    #[clap(short, long, default_value_t = 50000)]
    train_length: usize,

    #[clap(short, long, default_value_t = 1000)]
    mark_length: usize,

    #[clap(long, default_value_t = 10000)]
    test_length: usize,

    #[clap(long, default_value_t = 1)]
    epochs: usize,

    /// Evaluate every N training samples; 0 means only at the end of each epoch
    #[clap(long, default_value_t = 0)]
    verify_interval: usize,

    /// Number of distinct classification/assignment sets to evaluate at each verification checkpoint
    #[clap(long, default_value_t = 1)]
    classification_set_count: usize,

    /// Source split used for class-assignment/marking windows.
    #[clap(long, value_enum, default_value_t = MarkSplitArg::Validation)]
    mark_split: MarkSplitArg,

    /// Reuse the loaded training pool cyclically when the requested training budget exceeds 60k examples.
    #[clap(long, default_value_t = false)]
    cycle_train_pool: bool,

    #[clap(long, value_enum, default_value_t = DatasetArg::Fashion)]
    dataset: DatasetArg,

    #[clap(long, default_value = "data")]
    data_path: String,

    /// Spike probability per tick for max-intensity pixel.
    /// Reference: pixel/8 * intensity(2) Hz → 63.75Hz → 0.032/tick at dt=0.5ms.
    #[clap(short, long, default_value_t = 0.032)]
    poisson_rate: f32,

    /// Increment when re-presenting a sample with too few spikes.
    /// Reference: intensity += 1 → 31.875Hz → 0.016/tick at dt=0.5ms.
    #[clap(long, default_value_t = 0.016)]
    poisson_rate_inc: f32,

    #[clap(long, default_value_t = 0.007)]
    least_training_firing_rate: f32,

    /// Number of ticks to run per sample (reference: 350ms at dt=0.5ms = 700)
    #[clap(long, default_value_t = 700)]
    per_sample_ticks: usize,

    /// Number of zero-input ticks to run after each presentation attempt (reference: 150ms = 300)
    #[clap(long, default_value_t = 300)]
    rest_ticks: usize,

    #[clap(short, long, default_value_t = 400)]
    output_num: usize,

    #[clap(short, long, default_value_t = 1.0)]
    connection_rate: f32,

    #[clap(short, long, default_value_t = 0.0)]
    base_noise: f32,

    /// I->E conductance weight (positive; inhibition via reversal potential)
    #[clap(short, long, default_value_t = 17.0)]
    lateral_inhib_strength: f32,

    /// E->I conductance weight (positive)
    #[clap(long)]
    excitatory_inhibitory_strength: Option<f32>,

    #[clap(long, value_enum, default_value_t = NormalizationArg::L1)]
    normalization: NormalizationArg,

    /// Column-sum target for feedforward weight normalization
    #[clap(long, default_value_t = 78.0)]
    post_renorm_gain: f32,

    #[clap(long, default_value_t = 0.0)]
    slow_scaling_rate: f32,

    #[clap(long, default_value_t = 0.2)]
    slow_scaling_target_rate: f32,

    #[clap(long, default_value_t = 0.01)]
    slow_scaling_alpha: f32,

    #[clap(long, default_value_t = 0.25)]
    slow_scaling_min_gain: f32,

    #[clap(long, default_value_t = 4.0)]
    slow_scaling_max_gain: f32,

    #[clap(long)]
    feedforward_weight_max: Option<f32>,

    #[clap(long)]
    stdp_lr_plus: Option<f32>,

    #[clap(long)]
    stdp_lr_minus: Option<f32>,

    #[clap(long)]
    stdp_tau_plus: Option<f32>,

    #[clap(long)]
    stdp_tau_minus: Option<f32>,

    #[clap(long, default_value_t = 1e-3)]
    near_zero_weight_threshold: f32,

    #[clap(short, long, default_value_t = 0xdeadbeef19260817)]
    rng_seed: u64,

    /// Base path for checkpoint files. At each verification checkpoint, saves
    /// `{stem}_{samples}.{ext}` and overwrites `{stem}_latest.{ext}`.
    #[clap(long)]
    checkpoint_path: Option<PathBuf>,

    /// Resume training from an existing checkpoint file.
    #[clap(long)]
    resume_from: Option<PathBuf>,
}

fn fill_noise(bias: &mut [f32], base_noise: f32) {
    for b in bias.iter_mut() {
        *b = rand::random::<f32>() * base_noise;
    }
}

fn run_rest_phase(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    rates: &mut [f32],
    bias: &mut [f32],
    rest_ticks: usize,
    plasticity_enabled: bool,
) {
    if rest_ticks == 0 {
        return;
    }

    rates.fill(0.0);
    bias.fill(0.0);
    for _ in 0..rest_ticks {
        net.tick(rates, bias, plasticity_enabled, None);
    }
}

fn run_presentation(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    input: ndarray::ArrayView1<'_, f32>,
    rates: &mut [f32],
    bias: &mut [f32],
    fire_tracker: &mut [usize],
    poisson_rate: f32,
    poisson_rate_inc: f32,
    min_total_spikes: usize,
    per_sample_ticks: usize,
    rest_ticks: usize,
    base_noise: f32,
    plasticity_enabled: bool,
) -> f32 {
    let mut cur_poisson_rate = poisson_rate;

    loop {
        rates.iter_mut().zip(input.iter()).for_each(|(r, &v)| *r = v * cur_poisson_rate);
        fire_tracker.fill(0);

        for _ in 0..per_sample_ticks {
            fill_noise(bias, base_noise);
            net.tick(rates, bias, plasticity_enabled, Some(fire_tracker));
        }

        let total_fired: usize = fire_tracker.iter().sum();
        if total_fired >= min_total_spikes {
            if cur_poisson_rate > poisson_rate {
                eprintln!("sample needed rate escalation to {:.3} (fired {})", cur_poisson_rate, total_fired);
            }
            run_rest_phase(net, rates, bias, rest_ticks, plasticity_enabled);
            return cur_poisson_rate;
        }

        eprintln!("retry: rate {:.3} → fired {} (need {})", cur_poisson_rate, total_fired, min_total_spikes);

        run_rest_phase(net, rates, bias, rest_ticks, plasticity_enabled);

        cur_poisson_rate += poisson_rate_inc;
        if cur_poisson_rate > 1.0 {
            eprintln!("WARN: poisson_rate escalated to {:.2} without eliciting {} spikes — skipping sample",
                cur_poisson_rate, min_total_spikes);
            return cur_poisson_rate;
        }
    }
}

struct EvaluationSummary {
    accuracy: f32,
    avg_firing_rate: f32,
}

struct VerificationSummary {
    mean_accuracy: f32,
    min_accuracy: f32,
    max_accuracy: f32,
    mean_avg_firing_rate: f32,
}

fn verification_checkpoints_per_epoch(train_length: usize, verify_interval: usize) -> usize {
    if verify_interval == 0 {
        1
    } else {
        train_length.div_ceil(verify_interval)
    }
}

fn verification_mark_window(mark_length: usize, classification_set_count: usize, checkpoint_global_index: usize, classification_set_index: usize) -> (usize, usize) {
    let mark_window_index = checkpoint_global_index
        .checked_mul(classification_set_count)
        .and_then(|idx| idx.checked_add(classification_set_index))
        .expect("verification mark window index overflowed");
    let mark_start = mark_window_index
        .checked_mul(mark_length)
        .expect("verification mark start overflowed");
    let mark_end = mark_start.checked_add(mark_length).expect("verification mark end overflowed");
    (mark_start, mark_end)
}


fn train_pool_length(total_train_length: usize, cycle_train_pool: bool, validation_pool_length: usize) -> usize {
    if cycle_train_pool {
        total_train_length.min(MAX_TRAIN_AND_MARK_SAMPLES - validation_pool_length)
    } else {
        total_train_length
    }
}

fn validation_pool_length(total_mark_length: usize, mark_split: MarkSplitArg) -> usize {
    match mark_split {
        MarkSplitArg::Validation => total_mark_length,
        MarkSplitArg::Test => 0,
    }
}

fn test_pool_length(test_length: usize, total_mark_length: usize, mark_split: MarkSplitArg) -> usize {
    match mark_split {
        MarkSplitArg::Validation => test_length,
        MarkSplitArg::Test => test_length.max(total_mark_length),
    }
}

fn verification_checkpoint_number(epoch_progress: usize, train_length: usize, verify_interval: usize) -> usize {
    assert!(epoch_progress > 0 && epoch_progress <= train_length);
    if verify_interval == 0 {
        1
    } else {
        epoch_progress.div_ceil(verify_interval)
    }
}

fn checkpoint_path_with_suffix(base: &std::path::Path, suffix: &str) -> PathBuf {
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let new_name = match base.extension() {
        Some(ext) => format!("{}_{}.{}", stem, suffix, ext.to_string_lossy()),
        None => format!("{}_{}", stem, suffix),
    };
    base.with_file_name(new_name)
}

fn log_training_diagnostics(
    net: &snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    near_zero_weight_threshold: f32,
) {
    if let Some((min_gain, avg_gain, max_gain)) = net.feedforward_gain_stats() {
        println!("Feedforward gain statistics after training: min {:.4}, avg {:.4}, max {:.4}", min_gain, avg_gain, max_gain);
    }

    if let Some((min_rate, avg_rate, max_rate)) = net.slow_scaling_rate_ema_stats() {
        println!("Slow scaling EMA rate statistics after training: min {:.4}, avg {:.4}, max {:.4}", min_rate, avg_rate, max_rate);
    }

    let sparsity_stats = net.feedforward_sparsity_stats(near_zero_weight_threshold);
    println!(
        "Feedforward sparsity after training (|w| <= {:.4e} treated as near-zero): non-zero {}/{} ({:.2}%), present {}/{} ({:.2}%), near-zero among present {}/{} ({:.2}%)",
        near_zero_weight_threshold,
        sparsity_stats.nonzero_synapses(),
        sparsity_stats.total_slots,
        sparsity_stats.nonzero_ratio() * 100.0,
        sparsity_stats.present_synapses,
        sparsity_stats.total_slots,
        sparsity_stats.present_ratio() * 100.0,
        sparsity_stats.near_zero_synapses,
        sparsity_stats.present_synapses,
        sparsity_stats.near_zero_present_ratio() * 100.0,
    );
}

fn assign_classes(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    mark_data: ArrayView2<'_, f32>,
    mark_labels: ArrayView1<'_, u8>,
    rates: &mut [f32],
    bias: &mut [f32],
    fire_tracker: &mut [usize],
    args: &Args,
    min_total_spikes: usize,
    verbose: bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut prediction_matrix: Vec<Vec<usize>> = vec![vec![0; args.output_num]; 10];
    let mut class_cnt = vec![0usize; 10];
    for i in 0..mark_data.nrows() {
        let input = mark_data.slice(s![i, ..]);
        run_presentation(
            net,
            input,
            rates,
            bias,
            fire_tracker,
            args.poisson_rate,
            args.poisson_rate_inc,
            min_total_spikes,
            args.per_sample_ticks,
            args.rest_ticks,
            args.base_noise,
            false,
        );

        let tracker = prediction_matrix[mark_labels[i] as usize].as_mut_slice();
        tracker.iter_mut().zip(fire_tracker.iter()).for_each(|(dst, src)| *dst += *src);
        class_cnt[mark_labels[i] as usize] += 1;

        if verbose && (i + 1) % PROGRESS_INTERVAL == 0 {
            println!("Mark progress: {}/{}", i + 1, mark_data.nrows());
        }
    }

    let mut assigned_class = vec![0; args.output_num];
    let mut assigned_class_neurons = vec![0; 10];
    for i in 0..args.output_num {
        let average_fires = (0..10).map(|cls| {
            if class_cnt[cls] == 0 {
                0.0
            } else {
                prediction_matrix[cls][i] as f32 / class_cnt[cls] as f32
            }
        });
        let (predicted_class, r) = average_fires.enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        if verbose && r < 0.01 {
            println!("Warning: neuron {} has really low firing rate {}", i, r);
        }
        assigned_class[i] = predicted_class;
        assigned_class_neurons[predicted_class] += 1;
    }

    if verbose {
        println!("Assigned class neurons: {:?}", assigned_class_neurons);
    }
    if verbose && assigned_class_neurons.iter().any(|&c| c == 0) {
        println!("Warning: some classes have no assigned neurons");
    }

    (assigned_class, assigned_class_neurons)
}

fn evaluate_model(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    mark_data: ArrayView2<'_, f32>,
    mark_labels: ArrayView1<'_, u8>,
    test_data: ArrayView2<'_, f32>,
    test_labels: ArrayView1<'_, u8>,
    rates: &mut [f32],
    bias: &mut [f32],
    fire_tracker: &mut [usize],
    args: &Args,
    min_total_spikes: usize,
    verbose: bool,
) -> EvaluationSummary {
    let (assigned_class, assigned_class_neurons) = assign_classes(
        net,
        mark_data,
        mark_labels,
        rates,
        bias,
        fire_tracker,
        args,
        min_total_spikes,
        verbose,
    );

    let mut prediction_matrix = vec![vec![0; 10]; 10];
    let mut total_firing_cnt = vec![0usize; args.output_num];
    for i in 0..test_data.nrows() {
        let input = test_data.slice(s![i, ..]);
        run_presentation(
            net,
            input,
            rates,
            bias,
            fire_tracker,
            args.poisson_rate,
            args.poisson_rate_inc,
            min_total_spikes,
            args.per_sample_ticks,
            args.rest_ticks,
            args.base_noise,
            false,
        );

        let mut cls_firing_tot = [0; 10];
        for j in 0..args.output_num {
            total_firing_cnt[j] += fire_tracker[j];
            let cls = assigned_class[j];
            cls_firing_tot[cls] += fire_tracker[j];
        }

        let mut predicted_class = 0;
        let mut predicted_class_avg_firing = 0f32;
        for cls in 0..10 {
            if assigned_class_neurons[cls] == 0 {
                continue;
            }
            let avg_firing = cls_firing_tot[cls] as f32 / assigned_class_neurons[cls] as f32;
            if avg_firing > predicted_class_avg_firing {
                predicted_class = cls;
                predicted_class_avg_firing = avg_firing;
            }
        }

        prediction_matrix[test_labels[i] as usize][predicted_class] += 1;

        if verbose && (i + 1) % PROGRESS_INTERVAL == 0 {
            println!("Test progress: {}/{}", i + 1, test_data.nrows());
        }
    }

    let correct: usize = prediction_matrix.iter().enumerate().map(|(i, row)| row[i]).sum();
    let accuracy = correct as f32 / test_data.nrows() as f32;
    let max_firing_rate = total_firing_cnt.iter().cloned().max().unwrap() as f32 / test_data.nrows() as f32;
    let min_firing_rate = total_firing_cnt.iter().cloned().min().unwrap() as f32 / test_data.nrows() as f32;
    let avg_firing_rate = total_firing_cnt.iter().sum::<usize>() as f32 / (test_data.nrows() * args.output_num) as f32;

    if verbose {
        println!("Prediction matrix:");
        for row in prediction_matrix.iter() {
            for &count in row.iter() {
                print!("{:4} ", count);
            }
            println!();
        }

        println!("Accuracy: {:.2}%", accuracy * 100.0);
        println!("Firing rate statistics (per sample): max {:.2}, min {:.2}, avg {:.2}", max_firing_rate, min_firing_rate, avg_firing_rate);
        for bucket in 0..10 {
            let bucket_count = total_firing_cnt.iter().filter(|&&cnt| {
                let rate = cnt as f32 / test_data.nrows() as f32;
                rate >= bucket as f32 * 0.1 * max_firing_rate && rate < (bucket as f32 + 1.0) * 0.1 * max_firing_rate
            }).count();
            println!("Firing rate bucket [{:.2}, {:.2}): {} neurons", bucket as f32 * 0.1 * max_firing_rate, (bucket as f32 + 1.0) * 0.1 * max_firing_rate, bucket_count);
        }
    }

    EvaluationSummary { accuracy, avg_firing_rate }
}

fn evaluate_verification_checkpoint(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif>,
    mark_data: ArrayView2<'_, f32>,
    mark_labels: ArrayView1<'_, u8>,
    test_data: ArrayView2<'_, f32>,
    test_labels: ArrayView1<'_, u8>,
    rates: &mut [f32],
    bias: &mut [f32],
    fire_tracker: &mut [usize],
    args: &Args,
    min_total_spikes: usize,
    checkpoint_global_index: usize,
    verbose: bool,
) -> VerificationSummary {
    let runtime_state = net.snapshot_runtime_state();
    let mut accuracies = Vec::with_capacity(args.classification_set_count);
    let mut avg_firing_rates = Vec::with_capacity(args.classification_set_count);

    for classification_set_index in 0..args.classification_set_count {
        let (mark_start, mark_end) = verification_mark_window(
            args.mark_length,
            args.classification_set_count,
            checkpoint_global_index,
            classification_set_index,
        );
        net.restore_runtime_state(runtime_state.clone());

        let evaluation = evaluate_model(
            net,
            mark_data.slice(s![mark_start..mark_end, ..]),
            mark_labels.slice(s![mark_start..mark_end]),
            test_data,
            test_labels,
            rates,
            bias,
            fire_tracker,
            args,
            min_total_spikes,
            verbose && classification_set_index == 0,
        );
        println!(
            "Classification set {}/{} [{}..{}): accuracy {:.2}% avg firing {:.2}",
            classification_set_index + 1,
            args.classification_set_count,
            mark_start,
            mark_end,
            evaluation.accuracy * 100.0,
            evaluation.avg_firing_rate,
        );
        accuracies.push(evaluation.accuracy);
        avg_firing_rates.push(evaluation.avg_firing_rate);
    }

    net.restore_runtime_state(runtime_state);

    let min_accuracy = accuracies.iter().copied().fold(f32::INFINITY, f32::min);
    let max_accuracy = accuracies.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_accuracy = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    let mean_avg_firing_rate = avg_firing_rates.iter().sum::<f32>() / avg_firing_rates.len() as f32;

    VerificationSummary {
        mean_accuracy,
        min_accuracy,
        max_accuracy,
        mean_avg_firing_rate,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.epochs > 0);
    assert!(args.classification_set_count > 0);

    let run_configuration = checkpoint::RunConfiguration::from_args(&args);
    let checkpoint_base = args.checkpoint_path.clone();

    let total_train_length = args.train_length.checked_mul(args.epochs).expect("train_length * epochs overflowed");
    let checkpoints_per_epoch = verification_checkpoints_per_epoch(args.train_length, args.verify_interval);
    let total_verification_checkpoints = args.epochs.checked_mul(checkpoints_per_epoch).expect("epochs * verification checkpoints overflowed");
    let total_mark_windows = total_verification_checkpoints
        .checked_mul(args.classification_set_count)
        .expect("verification checkpoints * classification_set_count overflowed");
    let total_mark_length = args.mark_length.checked_mul(total_mark_windows).expect("mark_length * total mark windows overflowed");
    let validation_pool_length = validation_pool_length(total_mark_length, args.mark_split);
    let train_pool_length = train_pool_length(total_train_length, args.cycle_train_pool, validation_pool_length);
    let test_pool_length = test_pool_length(args.test_length, total_mark_length, args.mark_split);
    assert!(train_pool_length + validation_pool_length <= MAX_TRAIN_AND_MARK_SAMPLES,
        "Requested a training pool of {} samples and a {:?} marking pool of {} samples across {} verification windows, but only {} training/validation samples are available",
        train_pool_length,
        args.mark_split,
        validation_pool_length,
        total_mark_windows,
        MAX_TRAIN_AND_MARK_SAMPLES,
    );
    assert!(test_pool_length <= MAX_TEST_SAMPLES,
        "Requested a test pool of {} samples (test_length={}, mark_length_total={}, mark_split={:?}), but only {} test samples are available",
        test_pool_length,
        args.test_length,
        total_mark_length,
        args.mark_split,
        MAX_TEST_SAMPLES,
    );

    let feedforward_homeostasis = snn::network::FeedforwardHomeostasisConfig {
        normalization: args.normalization.into(),
        post_renorm_gain: args.post_renorm_gain,
        slow_scaling_rate: args.slow_scaling_rate,
        slow_scaling_target_rate: args.slow_scaling_target_rate,
        slow_scaling_alpha: args.slow_scaling_alpha,
        slow_scaling_min_gain: args.slow_scaling_min_gain,
        slow_scaling_max_gain: args.slow_scaling_max_gain,
    };

    let neuron_config = snn::network::MnistNeuronConfig::default();
    let stdp_config = snn::network::MnistStdpConfig {
        weight_max: args.feedforward_weight_max,
        lr_plus: args.stdp_lr_plus,
        lr_minus: args.stdp_lr_minus,
        tau_plus: args.stdp_tau_plus,
        tau_minus: args.stdp_tau_minus,
    };

    let mut resume_training_index = 0usize;
    let mut net = if let Some(resume_path) = args.resume_from.as_ref() {
        let training_checkpoint = checkpoint::load_checkpoint(resume_path)
            .with_context(|| format!("failed to load checkpoint from {}", resume_path.display()))?;
        if training_checkpoint.run_configuration != run_configuration {
            bail!(
                "checkpoint configuration does not match the current CLI arguments\ncurrent: {:?}\ncheckpoint: {:?}",
                run_configuration,
                training_checkpoint.run_configuration,
            );
        }
        if training_checkpoint.next_training_index > total_train_length {
            bail!(
                "checkpoint next_training_index {} exceeds requested training budget {}",
                training_checkpoint.next_training_index,
                total_train_length,
            );
        }
        resume_training_index = training_checkpoint.next_training_index;
        println!(
            "Resuming from checkpoint {} at training sample {}/{}",
            resume_path.display(),
            resume_training_index,
            total_train_length,
        );
        snn::network::PoissonInputNetwork::from_checkpoint(training_checkpoint.network)
    } else {
        let mut net = snn::network::new_mnist_network(
            args.output_num,
            args.connection_rate,
            args.lateral_inhib_strength,
            args.excitatory_inhibitory_strength,
            neuron_config,
            stdp_config,
            feedforward_homeostasis,
        );
        net.reset_neurons();
        net
    };

    if let Some(path) = checkpoint_base.as_ref() {
        println!("Checkpoint base path: {}", path.display());
    }
    if resume_training_index == total_train_length {
        println!("Checkpoint already contains the full training budget; nothing to do.");
        return Ok(());
    }

    println!("Loading {:?} dataset from {}", args.dataset, args.data_path);
    if args.cycle_train_pool && total_train_length > train_pool_length {
        println!(
            "Cycling training pool of {} samples to cover {} total training presentations",
            train_pool_length,
            total_train_length,
        );
    }
    println!("Using {:?} split for marking", args.mark_split);

    let mut dataset_builder = mnist::MnistBuilder::new();
    dataset_builder
        .base_path(&args.data_path)
        .label_format_digit()
        .training_set_length(train_pool_length as u32)
        .validation_set_length(validation_pool_length as u32)
        .test_set_length(test_pool_length as u32);
    let dataset = match args.dataset {
        DatasetArg::Mnist => dataset_builder.finalize(),
        DatasetArg::Fashion => dataset_builder.use_fashion_data().finalize(),
    };

    let train_data = ndarray::Array2::from_shape_vec((train_pool_length, 28 * 28), dataset.trn_img).unwrap().map(|e| *e as f32 / 255.0);
    // Don't care train labels
    let validation_data = ndarray::Array2::from_shape_vec((validation_pool_length, 28 * 28), dataset.val_img).unwrap().map(|e| *e as f32 / 255.0);
    let validation_labels = ndarray::Array1::from_shape_vec(validation_pool_length, dataset.val_lbl).unwrap();
    let test_pool_data = ndarray::Array2::from_shape_vec((test_pool_length, 28 * 28), dataset.tst_img).unwrap().map(|e| *e as f32 / 255.0);
    let test_pool_labels = ndarray::Array1::from_shape_vec(test_pool_length, dataset.tst_lbl).unwrap();
    let mark_data = match args.mark_split {
        MarkSplitArg::Validation => validation_data.view(),
        MarkSplitArg::Test => test_pool_data.slice(s![..total_mark_length, ..]),
    };
    let mark_labels = match args.mark_split {
        MarkSplitArg::Validation => validation_labels.view(),
        MarkSplitArg::Test => test_pool_labels.slice(s![..total_mark_length]),
    };
    let test_data = test_pool_data.slice(s![..args.test_length, ..]);
    let test_labels = test_pool_labels.slice(s![..args.test_length]);
    let mut bias = vec![0.0; args.output_num];
    let mut rates = vec![0.0; 28 * 28];

    let mut fire_tracker = vec![0usize; args.output_num];
    let min_total_spikes = ((args.least_training_firing_rate * args.per_sample_ticks as f32).ceil() as usize).max(1);

    // Build shuffled training index order from the deterministic RNG.
    let mut shuffle_rng = xoshiro256pp::Xoshiro256PlusPlus::new(args.rng_seed);
    let mut train_order: Vec<usize> = (0..train_data.nrows()).collect();
    shuffle_rng.shuffle(&mut train_order);

    // Replay additional shuffles for completed full passes so the RNG and
    // index order match the state they would have had in an uninterrupted run.
    let full_passes_completed = resume_training_index / train_data.nrows();
    for _ in 0..full_passes_completed {
        shuffle_rng.shuffle(&mut train_order);
    }

    let mut current_epoch = None;
    for i in resume_training_index..total_train_length {
        let epoch = i / args.train_length;
        let epoch_index = epoch + 1;
        let epoch_progress_before = i % args.train_length;

        if current_epoch != Some(epoch) {
            if epoch_progress_before == 0 {
                println!("Epoch {}/{} training start", epoch_index, args.epochs);
            } else {
                println!(
                    "Epoch {}/{} training resume at sample {}/{}",
                    epoch_index,
                    args.epochs,
                    epoch_progress_before + 1,
                    args.train_length,
                );
            }
            current_epoch = Some(epoch);
        }

        let pool_position = i % train_data.nrows();
        if pool_position == 0 && i > resume_training_index {
            shuffle_rng.shuffle(&mut train_order);
        }
        let train_pool_index = train_order[pool_position];
        let input = train_data.slice(s![train_pool_index, ..]);
        // Normalize feedforward weights before each training sample (reference: normalize_weights)
        net.normalize_all_feedforward_weights();
        let used_poisson_rate = run_presentation(
            &mut net,
            input,
            rates.as_mut_slice(),
            bias.as_mut_slice(),
            fire_tracker.as_mut_slice(),
            args.poisson_rate,
            args.poisson_rate_inc,
            min_total_spikes,
            args.per_sample_ticks,
            args.rest_ticks,
            args.base_noise,
            true,
        );

        if used_poisson_rate > args.poisson_rate {
            log::debug!("Increasing poisson rate to {} for sample {}", used_poisson_rate, i);
        }

        net.update_slow_synaptic_scaling(fire_tracker.as_slice(), args.per_sample_ticks);

        let epoch_progress = epoch_progress_before + 1;
        if epoch_progress % PROGRESS_INTERVAL == 0 {
            println!("Epoch {}/{} training progress: {}/{}", epoch_index, args.epochs, epoch_progress, args.train_length);
            // Diagnostic dump every progress interval
            let total_spikes: usize = fire_tracker.iter().sum();
            let max_spikes = *fire_tracker.iter().max().unwrap_or(&0);
            let active_neurons = fire_tracker.iter().filter(|&&f| f > 0).count();
            eprintln!("  SAMPLE_STATS: total_spikes={} max_single={} active_neurons={}/{} poisson_rate_used={:.3}",
                total_spikes, max_spikes, active_neurons, args.output_num, used_poisson_rate);
            net.dump_diagnostics(args.output_num);
        }

        let should_verify = if args.verify_interval == 0 {
            epoch_progress == args.train_length
        } else {
            epoch_progress % args.verify_interval == 0 || epoch_progress == args.train_length
        };

        if should_verify {
            let epoch_checkpoint_index = verification_checkpoint_number(epoch_progress, args.train_length, args.verify_interval);
            let checkpoint_global_index = epoch
                .checked_mul(checkpoints_per_epoch)
                .and_then(|idx| idx.checked_add(epoch_checkpoint_index - 1))
                .expect("global checkpoint index overflowed");
            let is_final_checkpoint = i + 1 == total_train_length;

            println!(
                "Epoch {}/{} verification checkpoint {}/{} after {}/{} training samples",
                epoch_index,
                args.epochs,
                epoch_checkpoint_index,
                checkpoints_per_epoch,
                epoch_progress,
                args.train_length,
            );
            log_training_diagnostics(&net, args.near_zero_weight_threshold);

            let verification = evaluate_verification_checkpoint(
                &mut net,
                mark_data,
                mark_labels,
                test_data,
                test_labels,
                rates.as_mut_slice(),
                bias.as_mut_slice(),
                fire_tracker.as_mut_slice(),
                &args,
                min_total_spikes,
                checkpoint_global_index,
                is_final_checkpoint,
            );
            println!(
                "Epoch {}/{} checkpoint {}/{} mean accuracy: {:.2}% (min {:.2}%, max {:.2}%), mean avg firing {:.2}",
                epoch_index,
                args.epochs,
                epoch_checkpoint_index,
                checkpoints_per_epoch,
                verification.mean_accuracy * 100.0,
                verification.min_accuracy * 100.0,
                verification.max_accuracy * 100.0,
                verification.mean_avg_firing_rate,
            );
        }

        let completed_training_samples = i + 1;
        let should_checkpoint = should_verify || completed_training_samples == total_train_length;
        if let Some(base) = checkpoint_base.as_ref() {
            if should_checkpoint {
                let training_checkpoint = checkpoint::TrainingCheckpoint::new(
                    run_configuration.clone(),
                    completed_training_samples,
                    net.to_checkpoint(),
                );
                let numbered_path = checkpoint_path_with_suffix(base, &completed_training_samples.to_string());
                checkpoint::save_checkpoint(&numbered_path, &training_checkpoint)
                    .with_context(|| format!("failed to save checkpoint to {}", numbered_path.display()))?;
                let latest_path = checkpoint_path_with_suffix(base, "latest");
                checkpoint::save_checkpoint(&latest_path, &training_checkpoint)
                    .with_context(|| format!("failed to save latest checkpoint to {}", latest_path.display()))?;
                println!(
                    "Saved checkpoint to {} at training sample {}/{}",
                    numbered_path.display(),
                    completed_training_samples,
                    total_train_length,
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::{Args, MarkSplitArg, MAX_TRAIN_AND_MARK_SAMPLES, test_pool_length, train_pool_length, validation_pool_length, verification_checkpoints_per_epoch, verification_mark_window};

    #[test]
    fn clap_configuration_is_valid() {
        Args::command().debug_assert();
    }

    #[test]
    fn verification_defaults_to_epoch_end_when_interval_is_zero() {
        assert_eq!(verification_checkpoints_per_epoch(2000, 0), 1);
    }

    #[test]
    fn verification_rounds_up_partial_interval() {
        assert_eq!(verification_checkpoints_per_epoch(2000, 750), 3);
    }

    #[test]
    fn verification_mark_windows_are_disjoint_across_sets_and_checkpoints() {
        assert_eq!(verification_mark_window(100, 3, 0, 0), (0, 100));
        assert_eq!(verification_mark_window(100, 3, 0, 2), (200, 300));
        assert_eq!(verification_mark_window(100, 3, 1, 0), (300, 400));
    }

    #[test]
    fn training_pool_can_cycle_over_full_mnist_training_set() {
        assert_eq!(train_pool_length(180_000, true, 0), MAX_TRAIN_AND_MARK_SAMPLES);
        assert_eq!(train_pool_length(180_000, true, 18_000), MAX_TRAIN_AND_MARK_SAMPLES - 18_000);
        assert_eq!(train_pool_length(20_000, true, 0), 20_000);
        assert_eq!(train_pool_length(20_000, false, 0), 20_000);
    }

    #[test]
    fn test_marking_uses_test_pool_budget() {
        assert_eq!(validation_pool_length(30_000, MarkSplitArg::Validation), 30_000);
        assert_eq!(validation_pool_length(30_000, MarkSplitArg::Test), 0);
        assert_eq!(test_pool_length(10_000, 10_000, MarkSplitArg::Validation), 10_000);
        assert_eq!(test_pool_length(10_000, 10_000, MarkSplitArg::Test), 10_000);
        assert_eq!(test_pool_length(2_000, 8_000, MarkSplitArg::Test), 8_000);
    }
}
