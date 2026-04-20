use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::snn::network::MnistNetworkCheckpoint;

const CHECKPOINT_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfiguration {
    pub train_length: usize,
    pub mark_length: usize,
    pub test_length: usize,
    pub epochs: usize,
    pub verify_interval: usize,
    pub classification_set_count: usize,
    pub mark_split: crate::MarkSplitArg,
    pub cycle_train_pool: bool,
    pub dataset: crate::DatasetArg,
    pub data_path: String,
    pub poisson_rate: f32,
    pub poisson_rate_inc: f32,
    pub least_training_firing_rate: f32,
    pub per_sample_ticks: usize,
    pub rest_ticks: usize,
    pub output_num: usize,
    pub connection_rate: f32,
    pub base_noise: f32,
    pub lateral_inhib_strength: f32,
    pub excitatory_inhibitory_strength: Option<f32>,
    pub neuron_tau: f32,
    pub neuron_threshold: f32,
    pub neuron_refractory_ticks: usize,
    pub threshold_homeostasis_tau: f32,
    pub threshold_homeostasis_inc: f32,
    pub inhibitory_neuron_tau: Option<f32>,
    pub inhibitory_neuron_threshold: Option<f32>,
    pub inhibitory_neuron_reset: Option<f32>,
    pub inhibitory_neuron_refractory_ticks: Option<usize>,
    pub inhibitory_threshold_homeostasis_tau: Option<f32>,
    pub inhibitory_threshold_homeostasis_inc: Option<f32>,
    pub feedforward_weight_max: Option<f32>,
    pub stdp_lr_plus: Option<f32>,
    pub stdp_lr_minus: Option<f32>,
    pub stdp_tau_plus: Option<f32>,
    pub stdp_tau_minus: Option<f32>,
    pub normalization: crate::NormalizationArg,
    pub post_renorm_gain: f32,
    pub slow_scaling_rate: f32,
    pub slow_scaling_target_rate: f32,
    pub slow_scaling_alpha: f32,
    pub slow_scaling_min_gain: f32,
    pub slow_scaling_max_gain: f32,
    pub near_zero_weight_threshold: f32,
    pub rng_seed: u64,
}

impl RunConfiguration {
    pub fn from_args(args: &crate::Args) -> Self {
        Self {
            train_length: args.train_length,
            mark_length: args.mark_length,
            test_length: args.test_length,
            epochs: args.epochs,
            verify_interval: args.verify_interval,
            classification_set_count: args.classification_set_count,
            mark_split: args.mark_split,
            cycle_train_pool: args.cycle_train_pool,
            dataset: args.dataset,
            data_path: args.data_path.clone(),
            poisson_rate: args.poisson_rate,
            poisson_rate_inc: args.poisson_rate_inc,
            least_training_firing_rate: args.least_training_firing_rate,
            per_sample_ticks: args.per_sample_ticks,
            rest_ticks: args.rest_ticks,
            output_num: args.output_num,
            connection_rate: args.connection_rate,
            base_noise: args.base_noise,
            lateral_inhib_strength: args.lateral_inhib_strength,
            excitatory_inhibitory_strength: args.excitatory_inhibitory_strength,
            neuron_tau: args.neuron_tau,
            neuron_threshold: args.neuron_threshold,
            neuron_refractory_ticks: args.neuron_refractory_ticks,
            threshold_homeostasis_tau: args.threshold_homeostasis_tau,
            threshold_homeostasis_inc: args.threshold_homeostasis_inc,
            inhibitory_neuron_tau: args.inhibitory_neuron_tau,
            inhibitory_neuron_threshold: args.inhibitory_neuron_threshold,
            inhibitory_neuron_reset: args.inhibitory_neuron_reset,
            inhibitory_neuron_refractory_ticks: args.inhibitory_neuron_refractory_ticks,
            inhibitory_threshold_homeostasis_tau: args.inhibitory_threshold_homeostasis_tau,
            inhibitory_threshold_homeostasis_inc: args.inhibitory_threshold_homeostasis_inc,
            feedforward_weight_max: args.feedforward_weight_max,
            stdp_lr_plus: args.stdp_lr_plus,
            stdp_lr_minus: args.stdp_lr_minus,
            stdp_tau_plus: args.stdp_tau_plus,
            stdp_tau_minus: args.stdp_tau_minus,
            normalization: args.normalization,
            post_renorm_gain: args.post_renorm_gain,
            slow_scaling_rate: args.slow_scaling_rate,
            slow_scaling_target_rate: args.slow_scaling_target_rate,
            slow_scaling_alpha: args.slow_scaling_alpha,
            slow_scaling_min_gain: args.slow_scaling_min_gain,
            slow_scaling_max_gain: args.slow_scaling_max_gain,
            near_zero_weight_threshold: args.near_zero_weight_threshold,
            rng_seed: args.rng_seed,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub version: u32,
    pub run_configuration: RunConfiguration,
    pub next_training_index: usize,
    pub network: MnistNetworkCheckpoint,
}

impl TrainingCheckpoint {
    pub fn new(run_configuration: RunConfiguration, next_training_index: usize, network: MnistNetworkCheckpoint) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            run_configuration,
            next_training_index,
            network,
        }
    }
}

pub fn save_checkpoint(path: &Path, checkpoint: &TrainingCheckpoint) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create checkpoint directory {}", parent.display()))?;
        }
    }

    let temp_file_name = path
        .file_name()
        .map(|name| format!("{}.tmp", name.to_string_lossy()))
        .unwrap_or_else(|| "checkpoint.tmp".to_string());
    let temp_path = path.with_file_name(temp_file_name);

    let file = File::create(&temp_path)
        .with_context(|| format!("failed to create checkpoint temp file {}", temp_path.display()))?;
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, checkpoint)
        .with_context(|| format!("failed to serialize checkpoint to {}", temp_path.display()))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush checkpoint temp file {}", temp_path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|err| err.into_error())
        .with_context(|| format!("failed to finish checkpoint temp file {}", temp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("failed to sync checkpoint temp file {}", temp_path.display()))?;

    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to move checkpoint temp file {} into place at {}",
            temp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<TrainingCheckpoint> {
    let file = File::open(path).with_context(|| format!("failed to open checkpoint {}", path.display()))?;
    let reader = BufReader::new(file);
    let checkpoint: TrainingCheckpoint = bincode::deserialize_from(reader)
        .with_context(|| format!("failed to deserialize checkpoint {}", path.display()))?;
    if checkpoint.version != CHECKPOINT_VERSION {
        bail!(
            "unsupported checkpoint version {} in {} (expected {})",
            checkpoint.version,
            path.display(),
            CHECKPOINT_VERSION,
        );
    }
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{RunConfiguration, TrainingCheckpoint, load_checkpoint, save_checkpoint};

    fn sample_run_configuration() -> RunConfiguration {
        RunConfiguration {
            train_length: 10,
            mark_length: 5,
            test_length: 4,
            epochs: 2,
            verify_interval: 5,
            classification_set_count: 1,
            mark_split: crate::MarkSplitArg::Validation,
            cycle_train_pool: true,
            dataset: crate::DatasetArg::Mnist,
            data_path: "data".to_string(),
            poisson_rate: 0.1,
            poisson_rate_inc: 0.05,
            least_training_firing_rate: 0.1,
            per_sample_ticks: 50,
            rest_ticks: 10,
            output_num: 4,
            connection_rate: 0.2,
            base_noise: 0.1,
            lateral_inhib_strength: -0.1,
            excitatory_inhibitory_strength: Some(0.05),
            neuron_tau: 50.0,
            neuron_threshold: 1.0,
            neuron_refractory_ticks: 0,
            threshold_homeostasis_tau: 5.0,
            threshold_homeostasis_inc: 2.0,
            inhibitory_neuron_tau: None,
            inhibitory_neuron_threshold: None,
            inhibitory_neuron_reset: None,
            inhibitory_neuron_refractory_ticks: None,
            inhibitory_threshold_homeostasis_tau: None,
            inhibitory_threshold_homeostasis_inc: None,
            feedforward_weight_max: None,
            stdp_lr_plus: None,
            stdp_lr_minus: None,
            stdp_tau_plus: None,
            stdp_tau_minus: None,
            normalization: crate::NormalizationArg::L1,
            post_renorm_gain: 1.0,
            slow_scaling_rate: 0.0,
            slow_scaling_target_rate: 0.2,
            slow_scaling_alpha: 0.01,
            slow_scaling_min_gain: 0.25,
            slow_scaling_max_gain: 4.0,
            near_zero_weight_threshold: 1e-3,
            rng_seed: 1234u64,
        }
    }

    #[test]
    fn checkpoint_file_roundtrip_preserves_payload() {
        let checkpoint = TrainingCheckpoint::new(
            sample_run_configuration(),
            12,
            crate::snn::network::new_mnist_network(
                4,
                0.2,
                -0.1,
                Some(0.05),
                crate::snn::network::MnistNeuronConfig::default(),
                crate::snn::network::MnistStdpConfig::default(),
                crate::snn::network::FeedforwardHomeostasisConfig {
                    normalization: crate::snn::network::FeedforwardNormalization::L1,
                    post_renorm_gain: 1.0,
                    slow_scaling_rate: 0.0,
                    slow_scaling_target_rate: 0.2,
                    slow_scaling_alpha: 0.01,
                    slow_scaling_min_gain: 0.25,
                    slow_scaling_max_gain: 4.0,
                },
            )
            .to_checkpoint(),
        );

        let unique_suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("stdp-experiments-checkpoint-{unique_suffix}.bin"));
        let _ = fs::remove_file(&path);

        save_checkpoint(&path, &checkpoint).unwrap();
        let loaded = load_checkpoint(&path).unwrap();

        assert_eq!(loaded, checkpoint);

        let _ = fs::remove_file(path);
    }
}