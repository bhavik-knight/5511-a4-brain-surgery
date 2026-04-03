"""Data generation for mechanistic interpretability of brain surgery models.

This module implements Q2 activation collection by running a fixed prompt
corpus through a ModelWrapper and saving token-aligned activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .model_wrapper import ModelWrapper
from .utils import ACTIVATIONS_DIR, ensure_dir_exists

type MetadataValue = int | str
type MetadataRow = dict[str, MetadataValue]


@dataclass(frozen=True)
class DatasetSummary:
    """Summary statistics for a generated activation dataset.

    Args:
            num_prompts: Total prompts processed.
            total_tokens: Total tokens captured across all prompts.
            average_seq_len: Average tokens per prompt.
            activation_shape: Final concatenated activation matrix shape.
    """

    num_prompts: int
    total_tokens: int
    average_seq_len: float
    activation_shape: tuple[int, int]


class DataGenerator:
    """Generate activation datasets from a prompt corpus.

    This class uses a ModelWrapper to capture residual-stream activations for
    each prompt, saves per-prompt artifacts via save_activations(), and builds
    a consolidated dataset tensor with token-level metadata.
    """

    def __init__(
        self,
        wrapper: ModelWrapper,
        *,
        batch_size: int = 4,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        """Initialize the data generator.

        Args:
                wrapper: ModelWrapper instance with hooks registered.
                batch_size: Prompts processed per batch. Kept small for VRAM safety.
                max_new_tokens: Max new tokens to generate per prompt.
                temperature: Sampling temperature for generation.
                top_p: Nucleus sampling threshold for generation.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.wrapper = wrapper
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def load_corpus(self) -> list[str]:
        """Load the soccer-domain prompt corpus (118 prompts).

        Returns:
                A list of prompt strings.
        """
        corpus_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "corpus"
            / "soccer_prompts.txt"
        )
        if corpus_path.exists():
            prompts = [
                line.strip()
                for line in corpus_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if prompts:
                return prompts

        return [
            # --- Rules & Roles ---
            "Who won the FIFA World Cup in 2022?",
            "Explain the offside rule in soccer in simple terms.",
            "What does a striker do in football?",
            "What is the role of a goalkeeper in soccer?",
            "Describe the job of a central defender in a football team.",
            "What does a midfielder do during a soccer match?",
            "Explain what a winger does in football.",
            "What is a penalty kick in soccer?",
            "What is a corner kick in football?",
            "What is a free kick in soccer?",
            "What happens when a player gets a red card in football?",
            "What happens when a player gets a yellow card in soccer?",
            "Explain what extra time means in football.",
            "What is a penalty shootout in soccer?",
            "What does possession mean in football?",
            "What is pressing in soccer tactics?",
            "Explain what counterattacking means in football.",
            "What is a formation in soccer?",
            "What does a 4-3-3 formation mean in football?",
            "What does a 4-4-2 formation mean in soccer?",
            "What is tiki-taka in football?",
            "What is a clean sheet in soccer?",
            "What is a hat-trick in football?",
            "What is an assist in soccer?",
            "What does it mean to dribble past a defender in football?",
            "What is a through ball in soccer?",
            "What is a cross in football?",
            "What is a volley in soccer?",
            "What is a header in football?",
            "What is VAR in soccer?",
            # --- Players & Clubs ---
            "Name three famous soccer players and describe them briefly.",
            "Who is Lionel Messi and why is he famous in football?",
            "Who is Cristiano Ronaldo and why is he important in soccer?",
            "Who is Kylian Mbappe in football?",
            "Who is Erling Haaland in soccer?",
            "What is FC Barcelona known for in football history?",
            "What is Real Madrid famous for in soccer?",
            "What is Manchester City known for in football?",
            "What is Liverpool known for in soccer history?",
            "What is the UEFA Champions League?",
            "What is the English Premier League?",
            "What is La Liga in football?",
            "What is the FIFA World Cup?",
            # --- Scenarios ---
            "Describe a dramatic last-minute goal in a soccer match.",
            "Describe a goalkeeper making an important save in football.",
            "Describe a team scoring from a counterattack in soccer.",
            "Describe a defender blocking a dangerous shot in football.",
            "Describe fans celebrating a goal in a soccer stadium.",
            "Describe a coach giving tactical instructions at halftime in football.",
            "Why is teamwork important in soccer?",
            # --- Rules & Fundamentals ---
            "What is the advantage rule in soccer?",
            "How large is a standard professional football pitch?",
            "Explain the role of a defensive midfielder.",
            "What does a full-back do in a standard soccer formation?",
            "Explain the 'false nine' tactical role.",
            "What happens if a match ends in a draw during a knockout tournament?",
            "Describe the responsibilities of the soccer referee.",
            "What are linesmen (assistant referees) responsible for?",
            "Why are studded cleats worn by soccer players?",
            "How long does a standard professional soccer match last?",
            "What is stoppage time or injury time in football?",
            "What constitutes a foul in the penalty box?",
            "Explain the concept of an indirect free kick.",
            "What happens after a ball goes out of bounds on the sideline?",
            # --- Teams & Rivalries ---
            (
                "Why is the rivalry between Real Madrid and FC Barcelona "
                "called El Clasico?"
            ),
            "What is the history of Manchester United Football Club?",
            "Describe the significance of Bayern Munich in European soccer.",
            "What is Arsenal Football Club famous for?",
            "Who are Juventus FC and what is their domestic dominance like?",
            "What is Paris Saint-Germain (PSG) known for in modern football?",
            "Describe the North London Derby between Arsenal and Tottenham.",
            "What is the significance of the Milan Derby (Derby della Madonnina)?",
            "Describe the history of Chelsea FC.",
            "Who has won the most English Premier League titles?",
            "What is the Serie A league in Italy known for tactically?",
            "Explain the history of the Bundesliga in Germany.",
            "What is the Copa Libertadores competition?",
            "Explain what makes the Copa America so competitive.",
            # --- Players & Icons ---
            "Who is Diego Maradona and why is he a football legend?",
            "Explain the legacy of Pele in the history of soccer.",
            "Who is Neymar Jr and what is his playing style?",
            "What made Zinedine Zidane one of the best midfielders?",
            "Describe the impact of Johan Cruyff on French and global football.",
            "Who is Ronaldinho and what makes his dribbling unique?",
            "Who is Kevin De Bruyne and what is his role at Manchester City?",
            "Describe the defensive prowess of Sergio Ramos.",
            "Who is Luka Modric and how does he orchestrate the midfield?",
            "Who was the top goalscorer in the 2018 World Cup?",
            "Describe Wayne Rooney's legendary career at Manchester United.",
            "What makes Virgil van Dijk a famously aggressive defender?",
            "Describe Manuel Neuer's impact as a modern center-back.",
            # --- Tactics & Analytical Scenarios ---
            "What is parked-the-bus meant to describe in football tactics?",
            "Describe the implementation of the high press.",
            "What does 'playing out from the back' entail?",
            "Why are inverted wingers used heavily in modern soccer?",
            "Describe the 'catenaccio' defensive tactical system used by Italy.",
            "What are the physical demands of playing box-to-box midfielder?",
            "Describe what it means to overlap as a full-back.",
            "How does a team effectively execute an offside trap?",
            "Analyze the importance of wing-backs in a 5-3-2 formation.",
            "What happens during the January transfer window?",
            "Explain how the away goals rule used to work in the Champions League.",
            "Describe the intense emotion of a UEFA Champions League Final.",
            "What happens physically to players during 120 minutes of football?",
            "How is youth development structured in European football academies?",
            "Describe the tactical adjustments a coach makes when reduced to 10 men.",
            "What is the 'Golden Boot' award in European football?",
            "What is the role of a sweeper in football?",
            "Explain the concept of a false full-back.",
            "What is a derby in soccer?",
            "How does relegation work in European football leagues?",
            "What is the difference between a red card and a second yellow card?",
            "What is a set piece in football?",
            "Describe the role of a playmaker in soccer.",
            "What is a loan move in the transfer market?",
            "Explain what a box-to-box midfielder does.",
            "What is a hat-trick of assists?",
            "Describe the atmosphere of a World Cup knockout match.",
        ]

    def generate_dataset(
        self,
        *,
        prompt_limit: int | None = None,
    ) -> tuple[Tensor, list[MetadataRow], DatasetSummary]:
        """Generate and save the activation dataset.

        Args:
                prompt_limit: Optional limit on number of prompts for smoke tests.

        Returns:
                A tuple with activation matrix, metadata list, and summary statistics.
        """
        prompts = self.load_corpus()
        if prompt_limit is not None:
            if prompt_limit <= 0:
                raise ValueError(
                    f"prompt_limit must be positive when provided, got {prompt_limit}"
                )
            prompts = prompts[:prompt_limit]
        else:
            if len(prompts) != 118:
                raise ValueError(
                    "Expected 118 prompts in corpus, got "
                    f"{len(prompts)}. Update load_corpus() to match the plan."
                )

        ensure_dir_exists(ACTIVATIONS_DIR)

        activation_chunks: list[Tensor] = []
        metadata: list[MetadataRow] = []
        total_tokens = 0
        seq_lens: list[int] = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch = prompts[batch_start : batch_start + self.batch_size]
            for idx, prompt in enumerate(batch):
                prompt_id = batch_start + idx
                _, activations = self.wrapper.generate_with_activations(
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                layer_acts = activations.get("layer")
                if layer_acts is None:
                    raise RuntimeError("Missing layer activations after generation")

                if layer_acts.dim() == 3:
                    if layer_acts.shape[0] != 1:
                        raise ValueError(
                            "Expected batch=1 activations, got "
                            f"{tuple(layer_acts.shape)}"
                        )
                    acts_2d = layer_acts[0]
                else:
                    acts_2d = layer_acts

                acts_2d = acts_2d.detach().cpu()
                activation_chunks.append(acts_2d)

                # Use internal token metadata from the wrapper for alignment.
                token_texts = self.wrapper._last_token_texts or []  # noqa: SLF001
                token_strs = self.wrapper._last_token_strs or []  # noqa: SLF001
                token_ids = []
                if self.wrapper._last_output_ids is not None:  # noqa: SLF001
                    token_ids = (
                        self.wrapper._last_output_ids[0].tolist()  # noqa: SLF001
                    )
                generated_text = self.wrapper._last_generated_text or ""  # noqa: SLF001

                seq_len = min(len(token_texts), acts_2d.shape[0])
                seq_lens.append(seq_len)
                total_tokens += seq_len
                hook_layer_index = (
                    self.wrapper.layer_idx if self.wrapper.layer_idx is not None else -1
                )
                hook_layer_name = f"model.model.layers[{hook_layer_index}]"

                for tok_idx in range(seq_len):
                    metadata.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_text": prompt,
                            "token_index": tok_idx,
                            "token_id": token_ids[tok_idx]
                            if tok_idx < len(token_ids)
                            else -1,
                            "token_text": token_texts[tok_idx],
                            "token_str": token_strs[tok_idx]
                            if tok_idx < len(token_strs)
                            else "",
                            "generated_text": generated_text,
                            "hook_layer_index": hook_layer_index,
                            "hook_layer_name": hook_layer_name,
                        }
                    )

                # Save per-prompt artifact using ModelWrapper helper.
                self.wrapper.save_activations(
                    save_dir=ACTIVATIONS_DIR,
                    batch_idx=prompt_id,
                    file_stem=f"soccer_prompt_{prompt_id:03d}",
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        activation_matrix = torch.cat(activation_chunks, dim=0)
        dataset_path = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
        torch.save(
            {
                "activation_matrix": activation_matrix,
                "metadata": metadata,
                "num_prompts": len(prompts),
                "total_tokens": total_tokens,
                "hidden_dim": activation_matrix.shape[1],
            },
            dataset_path,
        )

        average_seq_len = total_tokens / len(prompts)
        summary = DatasetSummary(
            num_prompts=len(prompts),
            total_tokens=total_tokens,
            average_seq_len=average_seq_len,
            activation_shape=tuple(activation_matrix.shape),
        )

        return activation_matrix, metadata, summary
