import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

###############################################################################
# 1) DEFINE THE MODELS (Generator, Teacher, Student)
###############################################################################
class Generator(nn.Module):
    """
    Generator: noise_dim -> hidden -> data_dim
    Produces synthetic samples (size = data_dim) from random noise (size = noise_dim).
    """
    def __init__(self, noise_dim, data_dim, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, z):
        return self.net(z)


class Teacher(nn.Module):
    """
    Teacher model: small MLP classifier with K real classes.
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(Teacher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def train_teacher(self, train_loader, lr=1e-4, epochs=10, device='cuda'):
        """
        Train the teacher model on its private data partition.
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.train()

        for epoch in range(epochs):
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self.forward(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    def predict(self, x):
        """
        Predict class labels (0..K-1).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds


class Student(nn.Module):
    """
    Student (Discriminator-like):
    - Has (K+1) outputs: K real classes + 1 'fake' class.
    """
    def __init__(self, input_dim, hidden_dim=64, num_classes=4):
        """
        num_classes should be K+1 if you have K real classes and 1 fake class.
        """
        super(Student, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        """
        Returns argmax among K+1 possible classes (0..K-1 are real, K is fake).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds

###############################################################################
# 2) HELPERS: Partition Data, Data Loader, Multi-class Noisy Aggregation
###############################################################################
def partition_data(X, y, n_teachers):
    """
    Splits the dataset into n_teachers disjoint subsets (roughly equal).
    Returns a list of (X_part, y_part) tuples.
    """
    data_splits = []
    N = len(X)
    split_size = N // n_teachers

    indices = np.random.permutation(N)
    for i in range(n_teachers):
        start = i * split_size
        end = (i + 1) * split_size if i < n_teachers - 1 else N
        idx = indices[start:end]
        data_splits.append((X[idx], y[idx]))
    return data_splits

def create_data_loader(X, y, batch_size=128):
    """
    Creates a PyTorch DataLoader for (X, y).
    """
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def noisy_aggregate_multiclass(teacher_ensemble, x_batch, num_classes, lap_scale=1.0, device='cpu'):
    """
    Multi-class aggregator with Laplace noise.
    - Each teacher predicts a label in [0..(num_classes-1)].
    - We tally the votes for each class, add Laplace noise, then pick argmax.

    Returns a tensor of shape (batch_size,) with aggregated labels in [0..(num_classes-1)].
    """
    x_batch = x_batch.to(device)
    batch_size = x_batch.size(0)

    # Collect teacher predictions
    all_votes = []
    for teacher in teacher_ensemble:
        teacher.eval()
        with torch.no_grad():
            logits = teacher(x_batch)
            preds = torch.argmax(logits, dim=1)  # shape: (batch_size,)
            all_votes.append(preds.cpu().numpy())  # store on CPU

    # Shape => (batch_size, n_teachers)
    all_votes = np.array(all_votes).T

    agg_labels = []
    for i in range(batch_size):
        votes_i = all_votes[i]  # teacher predictions for sample i
        # Count how many teachers voted for each class 0..(num_classes-1)
        counts = np.zeros(num_classes, dtype=np.float32)
        for c in votes_i:
            counts[c] += 1

        # Add Laplace noise to each class count
        noisy_counts = counts + np.random.laplace(loc=0.0, scale=lap_scale, size=num_classes)

        # Pick the class with the highest noisy count
        label_i = np.argmax(noisy_counts)
        agg_labels.append(label_i)

    return torch.tensor(agg_labels, dtype=torch.long, device=device)


###############################################################################
# 3) The Multi-Class PATE-GAN Training Loop
###############################################################################
def train_pate_gan_multiclass(generator,
                              student,
                              teacher_ensemble,
                              real_data_loader,
                              k_classes,     # The number of real classes = K
                              noise_dim,
                              num_epochs=100,
                              lap_scale=0.5,
                              lr=1e-4,
                              device='cpu'):
    """
    Multi-class PATE-GAN training:
      - Teacher ensemble has K real classes.
      - Student has (K+1) outputs (the extra one is 'fake').
      - Aggregator returns labels in [0..(K-1)] for real data.
      - We label fake data as class K.

    The Generator tries to produce data that is recognized as a random real class [0..K-1].
    """
    generator = generator.to(device)
    student = student.to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    s_optimizer = optim.Adam(student.parameters(), lr=lr/10)

    for epoch in range(num_epochs):
        for batch_idx, (real_data, _) in enumerate(real_data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # ========================
            # 1) Train Student
            # ========================
            s_optimizer.zero_grad()

            # (a) Real data => aggregator-labeled in [0..K-1]
            real_labels = noisy_aggregate_multiclass(
                teacher_ensemble, real_data, num_classes=k_classes,
                lap_scale=lap_scale, device=device
            )
            # Student forward => shape: (batch_size, K+1)
            logits_real = student(real_data)
            # Cross-entropy vs aggregator's label
            loss_real = F.cross_entropy(logits_real, real_labels)

            # (b) Fake data => label = K (the 'fake' class)
            z = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(z).detach()  # do not backprop through generator
            fake_labels = torch.full((batch_size,), fill_value=k_classes,
                                     dtype=torch.long, device=device)
            logits_fake = student(fake_data)
            loss_fake = F.cross_entropy(logits_fake, fake_labels)

            student_loss = loss_real + loss_fake
            student_loss.backward()
            s_optimizer.step()

            # ========================
            # 2) Train Generator
            # ========================
            g_optimizer.zero_grad()

            # We want the Student to classify generated samples as a "real" class [0..K-1].
            # We'll pick a random real class for each sample as the target.
            gen_target = torch.randint(low=0, high=k_classes, size=(batch_size,), device=device)

            z = torch.randn(batch_size, noise_dim, device=device)
            gen_data = generator(z)
            logits_gen = student(gen_data)  # shape: (batch_size, K+1)

            generator_loss = F.cross_entropy(logits_gen, gen_target)
            generator_loss.backward()
            g_optimizer.step()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs} Batch {batch_idx}] "
                      f"Student Loss: {student_loss.item():.4f}, "
                      f"Generator Loss: {generator_loss.item():.4f}")

    return generator, student


###############################################################################
# 4) MAIN FUNCTION WITH TSTR (Train on Synthetic, Test on Real)
###############################################################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------
    # 1. Load dataset from shuttle.tst
    #    The dataset is space-separated, has no header, and the last column is the target.
    # ------------------------------------------------------
    df = pd.read_csv("shuttle.tst", sep="\s+", header=None)  # Update the path if needed

    # Separate features and target (last column is target)
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    # Encode the target labels (even if already numeric, this ensures they are in the range 0..K-1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    X = features.values.astype(np.float32)

    # Number of distinct real classes
    K = len(label_encoder.classes_)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features, K={K} classes.")

    # ------------------------------------------------------
    # 2. Partition Data for multiple teachers
    # ------------------------------------------------------
    n_teachers = 4
    data_splits = partition_data(X, y, n_teachers=n_teachers)

    # ------------------------------------------------------
    # 3. Train each Teacher (K real classes)
    # ------------------------------------------------------
    teacher_ensemble = []
    for i, (X_part, y_part) in enumerate(data_splits):
        teacher = Teacher(input_dim=X.shape[1], hidden_dim=64, num_classes=K)
        part_loader = create_data_loader(X_part, y_part, batch_size=128)
        teacher.train_teacher(part_loader, lr=1e-3, epochs=5, device=device)
        teacher_ensemble.append(teacher)

    print(f"Trained {n_teachers} teachers on disjoint partitions.")

    # ------------------------------------------------------
    # 4. Create DataLoader for the entire dataset
    #    (We won't use y directly in the GAN loop; aggregator supplies labels)
    # ------------------------------------------------------
    full_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    real_data_loader = DataLoader(full_dataset, batch_size=128, shuffle=True)

    # ------------------------------------------------------
    # 5. Build Generator & Student
    #    - Teacher => K classes
    #    - Student => K+1 classes (the extra one is "fake")
    # ------------------------------------------------------
    noise_dim = 5
    generator = Generator(noise_dim=noise_dim, data_dim=X.shape[1], hidden_dim=64)
    student = Student(input_dim=X.shape[1], hidden_dim=64, num_classes=K+1)

    # ------------------------------------------------------
    # 6. Train the Multi-Class PATE-GAN
    # ------------------------------------------------------
    print("Starting Multi-Class PATE-GAN training...")
    trained_generator, trained_student = train_pate_gan_multiclass(
        generator=generator,
        student=student,
        teacher_ensemble=teacher_ensemble,
        real_data_loader=real_data_loader,
        k_classes=K,
        noise_dim=noise_dim,
        num_epochs=25,     # Increase as needed
        lap_scale=0.1,     # Laplace noise scale
        lr=1e-4,
        device=device
    )
    print("PATE-GAN training complete!")

    # ------------------------------------------------------
    # 7. Generate Synthetic Samples (TSTR approach)
    #    We'll produce as many synthetic samples as the real dataset size.
    # ------------------------------------------------------
    n_synth = X.shape[0]
    X_synth = []
    y_synth = []

    # Generate in mini-batches for efficiency
    batch_size = 128
    n_batches = (n_synth + batch_size - 1) // batch_size

    generator.eval()
    for b in range(n_batches):
        cur_size = min(batch_size, n_synth - b * batch_size)
        # Choose a random label in [0..K-1] for each synthetic sample
        gen_labels = torch.randint(low=0, high=K, size=(cur_size,), device=device)
        z = torch.randn(cur_size, noise_dim, device=device)

        with torch.no_grad():
            gen_data = generator(z)  # shape: (cur_size, input_dim)

        # Move to CPU numpy arrays
        gen_data_np = gen_data.cpu().numpy()
        gen_labels_np = gen_labels.cpu().numpy()

        X_synth.append(gen_data_np)
        y_synth.append(gen_labels_np)

    # Concatenate synthetic batches
    X_synth = np.concatenate(X_synth, axis=0)
    y_synth = np.concatenate(y_synth, axis=0)

    print(f"Generated {X_synth.shape[0]} synthetic samples.")

    # ------------------------------------------------------
    # 8. Save synthetic data to output.csv
    #    We'll write features + synthetic label in columns.
    # ------------------------------------------------------
    col_names = [f"feat_{i}" for i in range(X.shape[1])] + ["synthetic_label"]
    synthetic_data_df = pd.DataFrame(
        np.column_stack((X_synth, y_synth)),
        columns=col_names
    )
    synthetic_data_df.to_csv("output.csv", index=False)
    print("Saved synthetic data to output.csv.")

    # ------------------------------------------------------
    # 9. TSTR Evaluation: Train a classifier on synthetic data, test on real data
    # ------------------------------------------------------
    print("TSTR: Training Logistic Regression on synthetic data...")
    clf_synth = LogisticRegression(max_iter=200, multi_class='auto')
    clf_synth.fit(X_synth, y_synth)

    # Evaluate on real dataset
    print("Testing on real data (original labels):")
    real_preds = clf_synth.predict(X)  # shape: (N,)
    acc = accuracy_score(y, real_preds)
    print(f"TSTR Accuracy on real data: {acc:.4f}")

if __name__ == "__main__":
    main()
