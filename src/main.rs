use clap::Parser;
use ndarray::s;

mod snn;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum NormalizationArg {
    None,
    L1,
    L2,
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

#[derive(clap::Parser)]
struct Args {
    #[clap(short, long, default_value_t = 50000)]
    train_length: usize,

    #[clap(short, long, default_value_t = 1000)]
    mark_length: usize,

    #[clap(short, long, default_value_t = 10000)]
    test_length: usize,

    #[clap(short, long, default_value_t = 0.1)]
    poisson_rate: f32,

    #[clap(short, long, default_value_t = 0.05)]
    poisson_rate_inc: f32,

    #[clap(long, default_value_t = 0.1)]
    least_training_firing_rate: f32,

    /// Number of ticks to run per sample
    #[clap(short, long, default_value_t = 100)]
    per_sample_ticks: usize,

    #[clap(short, long, default_value_t = 1000)]
    output_num: usize,

    #[clap(short, long, default_value_t = 0.05)]
    connection_rate: f32,

    #[clap(short, long, default_value_t = 0.1)]
    base_noise: f32,

    #[clap(short, long, default_value_t = -0.1)]
    lateral_inhib_strength: f32,

    #[clap(long, value_enum, default_value_t = NormalizationArg::None)]
    normalization: NormalizationArg,

    #[clap(short, long, default_value_t = 0xdeadbeef19260817)]
    rng_seed: usize,
}

fn fill_noise(bias: &mut [f32], base_noise: f32) {
    for b in bias.iter_mut() {
        *b = rand::random::<f32>() * base_noise;
    }
}

fn run_presentation(
    net: &mut snn::network::PoissonInputNetwork<snn::neurons::Lif, snn::synapse::STDPSynapse>,
    input: ndarray::ArrayView1<'_, f32>,
    rates: &mut [f32],
    bias: &mut [f32],
    fire_tracker: &mut [usize],
    poisson_rate: f32,
    poisson_rate_inc: f32,
    min_total_spikes: usize,
    per_sample_ticks: usize,
    base_noise: f32,
    plasticity_enabled: bool,
) -> f32 {
    let mut cur_poisson_rate = poisson_rate;

    loop {
        rates.iter_mut().zip(input.iter()).for_each(|(r, &v)| *r = v * cur_poisson_rate);
        fire_tracker.fill(0);
        net.reset_neurons();

        for _ in 0..per_sample_ticks {
            fill_noise(bias, base_noise);
            net.tick(rates, bias, plasticity_enabled, Some(fire_tracker));
        }

        let total_fired: usize = fire_tracker.iter().sum();
        if total_fired >= min_total_spikes {
            return cur_poisson_rate;
        }

        cur_poisson_rate += poisson_rate_inc;
        assert!(cur_poisson_rate <= 1.0, "Unable to elicit enough spikes for a sample");
    }
}

fn main() {
    let args = Args::parse();

    let mut net = snn::network::new_mnist_network(
        args.output_num,
        args.connection_rate,
        args.lateral_inhib_strength,
        args.normalization.into(),
    );
    let dataset = mnist::MnistBuilder::new()
        .base_path("data")
        .use_fashion_data()
        .label_format_digit()
        .training_set_length(args.train_length as u32)
        .validation_set_length(args.mark_length as u32)
        .test_set_length(args.test_length as u32)
        .finalize();

    let train_data = ndarray::Array2::from_shape_vec((args.train_length, 28 * 28), dataset.trn_img).unwrap().map(|e| *e as f32 / 255.0);
    // Don't care train labels
    let mark_data = ndarray::Array2::from_shape_vec((args.mark_length, 28 * 28), dataset.val_img).unwrap().map(|e| *e as f32 / 255.0);
    let mark_labels = ndarray::Array1::from_shape_vec(args.mark_length, dataset.val_lbl).unwrap();
    let test_data = ndarray::Array2::from_shape_vec((args.test_length, 28 * 28), dataset.tst_img).unwrap().map(|e| *e as f32 / 255.0);
    let test_labels = ndarray::Array1::from_shape_vec(args.test_length, dataset.tst_lbl).unwrap();
    let mut bias = vec![0.0; args.output_num];
    let mut rates = vec![0.0; 28 * 28];

    let mut fire_tracker = vec![0usize; args.output_num];
    let min_total_spikes = ((args.least_training_firing_rate * args.per_sample_ticks as f32).ceil() as usize).max(1);

    // Run training
    const PROGRESS_INTERVAL: usize = 100;
    for i in 0..args.train_length {
        let input = train_data.slice(s![i, ..]);
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
            args.base_noise,
            true,
        );

        if used_poisson_rate > args.poisson_rate {
            println!("Increasing poisson rate to {} for sample {}", used_poisson_rate, i);
        }

        if (i + 1) % PROGRESS_INTERVAL == 0 {
            println!("Training progress: {}/{}", i + 1, args.train_length);
        }
    }

    // Run validation, get prediction mapping
    let mut prediction_matrix: Vec<Vec<usize>> = vec![vec![0; args.output_num]; 10];
    let mut class_cnt = vec![0usize; 10];
    for i in 0..args.mark_length {
        let input = mark_data.slice(s![i, ..]);
        run_presentation(
            &mut net,
            input,
            rates.as_mut_slice(),
            bias.as_mut_slice(),
            fire_tracker.as_mut_slice(),
            args.poisson_rate,
            args.poisson_rate_inc,
            min_total_spikes,
            args.per_sample_ticks,
            args.base_noise,
            false,
        );

        let tracker = prediction_matrix[mark_labels[i] as usize].as_mut_slice();
        tracker.iter_mut().zip(fire_tracker.iter()).for_each(|(dst, src)| *dst += *src);

        // Get the predicted class
        class_cnt[mark_labels[i] as usize] += 1;

        if (i + 1) % PROGRESS_INTERVAL == 0 {
            println!("Mark progress: {}/{}", i + 1, args.mark_length);
        }
    }

    // Neuron id -> prediction
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
        if r < 0.01 {
            println!("Warning: neuron {} has really low firing rate {}", i, r);
        }
        assigned_class[i] = predicted_class;
        assigned_class_neurons[predicted_class] += 1;
    }

    println!("Assigned class neurons: {:?}", assigned_class_neurons);
    if assigned_class_neurons.iter().any(|&c| c == 0) {
        println!("Warning: some classes have no assigned neurons");
    }
    
    // Run testing
    prediction_matrix = vec![vec![0; 10]; 10];
    let mut total_firing_cnt = vec![0usize; args.output_num];
    for i in 0..args.test_length {
        let input = test_data.slice(s![i, ..]);
        run_presentation(
            &mut net,
            input,
            rates.as_mut_slice(),
            bias.as_mut_slice(),
            fire_tracker.as_mut_slice(),
            args.poisson_rate,
            args.poisson_rate_inc,
            min_total_spikes,
            args.per_sample_ticks,
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
        let mut predicted_class_avg_firing= 0f32;
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

        // Average firing rate by assigned classes

        prediction_matrix[test_labels[i] as usize][predicted_class] += 1;

        if (i + 1) % PROGRESS_INTERVAL == 0 {
            println!("Test progress: {}/{}", i + 1, args.test_length);
        }
    }

    // Print prediction matrix
    println!("Prediction matrix:");
    for row in prediction_matrix.iter() {
        for &count in row.iter() {
            print!("{:4} ", count);
        }
        println!();
    }

    let correct: usize = prediction_matrix.iter().enumerate().map(|(i, row)| row[i]).sum();
    let accuracy = correct as f32 / args.test_length as f32;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Compute firing rate statistics
    let max_firing_rate = total_firing_cnt.iter().cloned().max().unwrap() as f32 / args.test_length as f32;
    let min_firing_rate = total_firing_cnt.iter().cloned().min().unwrap() as f32 / args.test_length as f32;
    let avg_firing_rate = total_firing_cnt.iter().sum::<usize>() as f32 / (args.test_length * args.output_num) as f32;
    println!("Firing rate statistics (per sample): max {:.2}, min {:.2}, avg {:.2}", max_firing_rate, min_firing_rate, avg_firing_rate);
    // Output in 10 buckets
    for bucket in 0..10 {
        let bucket_count = total_firing_cnt.iter().filter(|&&cnt| {
            let rate = cnt as f32 / args.test_length as f32;
            rate >= bucket as f32 * 0.1 * max_firing_rate && rate < (bucket as f32 + 1.0) * 0.1 * max_firing_rate
        }).count();
        println!("Firing rate bucket [{:.2}, {:.2}): {} neurons", bucket as f32 * 0.1 * max_firing_rate, (bucket as f32 + 1.0) * 0.1 * max_firing_rate, bucket_count);
    }
}
