import uuid
from pathlib import Path
from typing import List, Optional, Any, Dict, Set
from datetime import datetime, timezone
from contextlib import contextmanager
import numpy as np

from Denabase.db.store import FileStore
from Denabase.db.schema import (
    EntryRecord, EntryMeta, DBMeta, 
    ProvenanceRecord, EncodingRecipeRecord, VerificationRecord
)
from Denabase.embed.feature_embedder import FeatureEmbedder
from Denabase.embed.index import VectorIndex
from Denabase.core.cache import DiskCache
from Denabase.ir import compile_ir, normalize_ir, IR
from Denabase.cnf.cnf_types import CnfDocument
from Denabase.trace import EncodingTrace
from Denabase.cnf.cnf_stats import compute_cnf_stats
from Denabase.profile.cnf_profile import compute_cnf_profile, compute_ir_profile
from Denabase.graph.fingerprints import compute_fingerprint, Fingerprint
from Denabase.verify.verifier import CnfVerifier, VerificationConfig
from Denabase.profile.profile_types import ConstraintProfile
from Denabase.nl import NLEmbedder, DEFAULT_NL_DIM
from Denabase.alpha import AlphaModel, AlphaExample, extract_alpha_features
from Denabase.retrieval import fuse_scores, mmr_select, clamp_alpha
from Denabase.core.cache import DiskCache
from Denabase.cnf.cnf_io import read_dimacs_from_string

def entry_dedupe_key(meta: EntryMeta, hashes: Dict[str, str]) -> str:
    """Returns a stable key for deduplicating entries."""
    if meta.dataset_id:
        return f"satbench:{meta.dataset_id}"
    if meta.problem_id:
        return f"pid:{meta.problem_id}"
    return f"cnf:{hashes.get('content_hash', 'unknown')}"

class DenaBase:
    """
    Main interface for the persistent database of certified cases.
    """
    
    @staticmethod
    def create(root: str) -> "DenaBase":
        """Creates or opens a database at root."""
        return DenaBase(root)

    @staticmethod
    def open(root: str) -> "DenaBase":
        """Opens an existing database at root."""
        return DenaBase(root)

    @property
    def db_meta(self) -> DBMeta:
        """Loads and returns global database metadata."""
        return self.store.get_meta()

    @property
    def total_entries(self) -> int:
        """Returns the total number of entries in the database."""
        return self.db_meta.counts.get("entries", 0)

    def __init__(self, root: str):
        self.root = Path(root)
        self.store = FileStore(self.root)
        
        # Load embedder and index
        self.embedder_path = self.store.dirs["indexes"] / "embedder.joblib"
        if self.embedder_path.exists():
            self.embedder = FeatureEmbedder.load(self.embedder_path)
        else:
            self.embedder = FeatureEmbedder()
            
        self.index_path = self.store.dirs["indexes"] / "vector_index.bin"
        self.cache_dir = self.store.root / "cache"
        self.index = VectorIndex(cache_dir=self.cache_dir)
        
        if self.index_path.exists():
             self.index = VectorIndex.load(self.index_path)
             # Reload doesn't carry config like cache_dir usually unless saved
             # But VectorIndex.load is static.
             # We need to re-attach cache.
             self.index.cache = self.index.cache or DiskCache(self.cache_dir)
             
        # NL Embedder & Index
        self.nl_embedder_path = self.store.dirs["indexes"] / "nl_embedder.joblib"
        self.nl_index_path = self.store.dirs["indexes"] / "nl_vector_index.bin"
        
        if self.nl_embedder_path.exists():
            self.nl_embedder = NLEmbedder.load(self.nl_embedder_path)
        else:
            self.nl_embedder = NLEmbedder()
            
        self.nl_index = VectorIndex(cache_dir=self.cache_dir / "nl")
        if self.nl_index_path.exists():
            self.nl_index = VectorIndex.load(self.nl_index_path)
            self.nl_index.cache = self.nl_index.cache or DiskCache(self.cache_dir / "nl")

        # Alpha Model
        self.alpha_model_path = self.store.dirs["indexes"] / "alpha_model.joblib"
        if self.alpha_model_path.exists():
            self.alpha_model = AlphaModel.load(self.alpha_model_path)
        else:
            self.alpha_model = AlphaModel()

        self._in_bulk_mode = False

    def begin_bulk_ingest(self):
        """Enable bulk mode: defer index rebuilds and atomic writes."""
        self._in_bulk_mode = True
        self.store.begin_bulk()

    def end_bulk_ingest(self, rebuild: bool = True):
        """Disable bulk mode and optionally rebuild indexes."""
        if not self._in_bulk_mode: return
        
        self.store.end_bulk()
        self._in_bulk_mode = False
        
        if rebuild:
            meta = self.db_meta
            if meta.structural_index_dirty:
                self.rebuild_index()
            if meta.nl_index_dirty:
                self.rebuild_nl_index()
            if meta.alpha_model_dirty:
                self.rebuild_alpha_model()

    @contextmanager
    def bulk_ingest(self, rebuild: bool = True):
        """Context manager for bulk ingestion."""
        self.begin_bulk_ingest()
        try:
            yield
        finally:
            self.end_bulk_ingest(rebuild=rebuild)

    def _extract_profile_tokens(self, fp: Fingerprint) -> List[str]:
        """Generates simple tokens for inverted index."""
        tokens = []
        tokens.append(f"sig:{fp.signature_key}")
        # Add basic profile buckets?
        # e.g. from feature vector or raw profile
        # For now just use signature key and maybe graph hash
        tokens.append(f"wl:{fp.wl_hash}")
        return tokens

    def add_ir(self, 
               ir_obj: Any, 
               family: str, 
               problem_id: str, 
               meta: Dict[str, Any] = None,
               verify: bool = True,
               **entry_kwargs) -> str:
        """
        Full pipeline: IR -> CNF -> Profile/Fingerprint -> Verify -> Store.
        entry_kwargs matches EntryMeta fields (e.g. source, nl_text).
        """
        # 1. Normalize
        norm_ir = normalize_ir(ir_obj)
        
        # 2. Compile (Standard) - Encoding Selection could go here
        clauses, varmap = compile_ir(norm_ir)
        # Compute num_vars correctly
        max_var = len(varmap)
        if clauses:
            max_c = max(abs(l) for c in clauses for l in c)
            max_var = max(max_var, max_c)
            
        doc = CnfDocument(num_vars=max_var, clauses=clauses)
        
        # 3. Profile & Fingerprint
        cnf_prof = compute_cnf_profile(doc)
        ir_prof = compute_ir_profile(norm_ir)
        
        # Merge: prefer IR counts/cards, keep CNF stats
        prof = cnf_prof.model_copy()
        prof.counts.update(ir_prof.counts)
        prof.cardinalities.update(ir_prof.cardinalities)
        
        fp = compute_fingerprint(doc, prof)
        
        # 4. Verify
        verification_rec = None
        if verify:
            # Strict verification config
            config = VerificationConfig(
                seconds_max=2.0,
                num_metamorphic=10,
                check_simplify_equisat=True,
                solver_calls_max=50
            )
            verifier = CnfVerifier(config)
            res = verifier.verify(doc)
            
            verification_rec = VerificationRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                config=config.model_dump(),
                passed=(res.outcome == "PASSED"),
                checks_run=res.checks_run,
                failures=res.failures,
                stats={"outcome": res.outcome}
            )
            
            if res.outcome != "PASSED":
                from Denabase.core.errors import VerificationError
                failures = "; ".join(res.failures)
                raise VerificationError(f"Verification failed for IR entry: {failures}")
            
        # 5. Embed
        # Support unfitted
        emb_vec = self.embedder.embed(doc, prof)
        
        # 6. Prepare Records
        eid = str(uuid.uuid4())
        
        entry_meta = EntryMeta(
            entry_id=eid,
            family=family,
            problem_id=problem_id,
            user_meta=meta or {},
            **entry_kwargs
        )
        
        record = EntryRecord(
            id=eid,
            meta=entry_meta,
            stats_summary={"n_vars": doc.num_vars, "n_clauses": len(doc.clauses)},
            hashes={"content_hash": fp.content_hash, "wl_hash": fp.wl_hash}
        )
        
        artifacts = {
            "ir": norm_ir, # Serializing raw IR if JSON serializable lists/dicts
            "cnf": f"p cnf {doc.num_vars} {len(doc.clauses)}\n" + "\n".join(" ".join(map(str, c)) + " 0" for c in doc.clauses),
            "profile": prof,
            "fingerprint": fp,
            "provenance": ProvenanceRecord(gadgets=[])
        }
        
        # 7. Store
        self.store.put_entry(record, artifacts, verification_record=verification_rec)
        
        # 8. Update Indexes
        tokens = self._extract_profile_tokens(fp)
        self.store.update_indexes(eid, fp.signature_key, tokens)
        self.index.add([emb_vec.tolist()], [eid])
        
        if self._in_bulk_mode:
            meta = self.db_meta
            meta.structural_index_dirty = True
            meta.nl_index_dirty = True
            meta.alpha_model_dirty = True
            self.store.set_meta(meta)
        else:
            self.index.save(self.index_path)
            self._update_ml_components()
        
        return eid

    def add_cnf(self, 
                doc: CnfDocument, 
                family: str, 
                problem_id: str, 
                meta: Dict[str, Any] = None,
                verify: bool = False,
                **entry_kwargs) -> str:
        """
        Adds a raw CNF document.
        If verify=True, runs metamorphic verification before ingesting.
        entry_kwargs matches EntryMeta fields (e.g. source, nl_text).
        """
        # 1. Verify First (if requested)
        verification_rec = None
        if verify:
            config = VerificationConfig(
                seconds_max=2.0,
                num_metamorphic=10,
                check_simplify_equisat=True,
                solver_calls_max=50
            )
            verifier = CnfVerifier(config)
            res = verifier.verify(doc)
            
            verification_rec = VerificationRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                config=config.model_dump(),
                passed=(res.outcome == "PASSED"),
                checks_run=res.checks_run,
                failures=res.failures,
                stats={"outcome": res.outcome}
            )
            
            if res.outcome != "PASSED":
                from Denabase.core.errors import VerificationError
                failures = "; ".join(res.failures)
                raise VerificationError(f"Verification failed for CNF entry: {failures}")

        prof = compute_cnf_profile(doc)
        fp = compute_fingerprint(doc, prof)
        emb_vec = self.embedder.embed(doc, prof)
        
        eid = str(uuid.uuid4())
        
        entry_meta = EntryMeta(
            entry_id=eid,
            family=family,
            problem_id=problem_id,
            user_meta=meta or {},
            **entry_kwargs
        )
        
        record = EntryRecord(
            id=eid,
            meta=entry_meta,
            stats_summary={"n_vars": doc.num_vars, "n_clauses": len(doc.clauses)},
            hashes={"content_hash": fp.content_hash, "wl_hash": fp.wl_hash}
        )
        
        cnf_str = f"p cnf {doc.num_vars} {len(doc.clauses)}\n" + "\n".join(" ".join(map(str, c)) + " 0" for c in doc.clauses)
        
        artifacts = {
            "cnf": cnf_str,
            "profile": prof,
            "fingerprint": fp
        }
        
        self.store.put_entry(record, artifacts, verification_record=verification_rec)
        
        tokens = self._extract_profile_tokens(fp)
        self.store.update_indexes(eid, fp.signature_key, tokens)
        self.index.add([emb_vec.tolist()], [eid])
        
        if self._in_bulk_mode:
            meta = self.db_meta
            meta.structural_index_dirty = True
            meta.nl_index_dirty = True
            meta.alpha_model_dirty = True
            self.store.set_meta(meta)
        else:
            self.index.save(self.index_path)
            self._update_ml_components()
        
        return eid

    def _update_ml_components(self):
        """Updates NL index and Alpha model."""
        meta = DBMeta(**self.store._read_json(self.store.meta_file))
        total = meta.counts.get("entries", 0)
        
        # 1. Update NL Index (<= 2000 entries: refit all)
        if total <= 2000:
            self.rebuild_nl_index()
            
        # 2. Update Alpha Model (>= 20 entries)
        min_ex = 20
        import os
        if "DENABASE_ALPHA_MIN_EXAMPLES" in os.environ:
             try: min_ex = int(os.environ["DENABASE_ALPHA_MIN_EXAMPLES"])
             except: pass
             
        if total >= min_ex and meta.alpha_model_enabled:
            self.rebuild_alpha_model()


    def add_satbench_case(self,
                          dataset_id: str,
                          family: str,
                          problem_id: str,
                          nl_text: str,
                          expected_label: Any,
                          split: str = None,
                          cnf_doc: CnfDocument = None,
                          cnf_path: str = None,
                          cnf_dimacs: str = None,
                          tags: List[str] = None,
                          user_meta: Dict[str, Any] = None,
                          verify: bool = False) -> str:
        """
        Specialized ingestion for SAT-Bench cases with NL and metadata.
        """
        from Denabase.db.schema import canonical_label
        
        # 1. Resolve CNF
        if cnf_doc:
            doc = cnf_doc
        elif cnf_path:
             from Denabase.cnf.cnf_io import load_cnf
             doc = load_cnf(Path(cnf_path))
        elif cnf_dimacs:
             doc = read_dimacs_from_string(cnf_dimacs)
        else:
             raise ValueError("Must provide cnf_doc, cnf_path, or cnf_dimacs")

        # 2. Canonicalize
        lbl = canonical_label(expected_label)
        
        # 3. Add CNF with params
        # This will index structural tokens
        eid = self.add_cnf(
            doc, 
            family=family, 
            problem_id=problem_id, 
            meta=user_meta, 
            verify=verify,
            # Extra fields
            source="satbench",
            nl_text=nl_text,
            expected_label=lbl,
            split=split,
            dataset_id=dataset_id,
            tags=tags or []
        )
        
        # 4. Add NL/Meta tokens to index
        # add_cnf calls update_indexes with structural tokens.
        # We append more tokens.
        extra_tokens = []
        
        # Label/Split
        if lbl: extra_tokens.append(f"label:{lbl}")
        if split: extra_tokens.append(f"split:{split}")
        extra_tokens.append(f"family:{family}") # Explicit family token?
        
        # NL Bag of words
        if nl_text:
            import re
            # Simple safe tokenizer
            words = re.findall(r"[a-zA-Z0-9]+", nl_text.lower())
            # Filter
            valid_words = [w for w in words if 3 <= len(w) <= 20]
            # Cap at 200
            valid_words = valid_words[:200]
            # Prefix
            nl_tokens = [f"nl:{w}" for w in valid_words]
            extra_tokens.extend(nl_tokens)
            
        if extra_tokens:
            # We don't have signature locally, can recompute or just pass None for sig?
            # store.update_indexes appends eid to sig list if signature_key provided.
            # If we pass a dummy signature key, it will create a dummy entry.
            # We want to update PROFILE TOKENS.
            # store.update_indexes(eid, signature_key, tokens)
            # If we don't restart fp computation, we can't update signature index effectively unless we know sig key.
            # However, signature index was ALREADY updated by add_cnf.
            # We only want to update tokens.
            # We can use a dummy signature key? No, that pollutes.
            # Store API requires signature_key.
            
            # Allow store.update_indexes to handle None signature?
            # Let's check store.py. 
            # if signature_key not in sigs: sigs[signature_key] = []
            
            # We should probably get the fingerprint from doc again to match key?
            # Or make add_cnf return (eid, fp)?
            # Changing return signature of add_cnf breaks interface.
            
            # Simplest: Recompute fingerprint (lightweight compared to run).
            # ACTUALLY, verify=True might be slow, but fp is fast.
            # Recomputing FP:
            from Denabase.profile.cnf_profile import compute_cnf_profile
            from Denabase.graph.fingerprints import compute_fingerprint
            
            # Avoid full embedded logic in add_cnf again?
            # Yes, recomputing FP locally is fine.
            prof = compute_cnf_profile(doc)
            fp = compute_fingerprint(doc, prof)
            
            self.store.update_indexes(eid, fp.signature_key, extra_tokens)
            
        return eid

    def query_similar(self, 
                      query_obj: Any, 
                      topk: int = 5,
                      alpha: Optional[float] = None,
                      nl_query_text: Optional[str] = None,
                      use_learned_alpha: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves similar entries using both structural and NL similarity.
        Fuses scores using alpha (trust structure).
        """
        # A. Process Query Structure
        struct_scores = {}
        has_struct = False
        if query_obj is not None:
            if isinstance(query_obj, CnfDocument):
                doc = query_obj
            else:
                 # Assume IR
                 norm = normalize_ir(query_obj)
                 clauses, varmap = compile_ir(norm)
                 
                 max_var = len(varmap)
                 if clauses:
                     max_c = max(abs(l) for c in clauses for l in c)
                     max_var = max(max_var, max_c)
                 
                 doc = CnfDocument(num_vars=max_var, clauses=clauses)
                 
            prof = compute_cnf_profile(doc)
            fp = compute_fingerprint(doc, prof)
            q_vec = self.embedder.embed(doc, prof)
            
            # B. Candidate Prefilter (Structural)
            cand_ids = set(self.store.get_candidate_ids_by_signature(fp.signature_key))
            
            # Structural Query
            # Combine signature matches + top-N vector neighbors
            v_res = self.index.query(q_vec.tolist(), k=topk*2)
            for eid, score in v_res:
                struct_scores[eid] = score
                
            for eid in cand_ids:
                if eid not in struct_scores:
                    # Signature match but not in top-N vectors (optional weight)
                    struct_scores[eid] = 0.5
            has_struct = True
        
        # C. NL Query
        nl_scores = {}
        has_nl_query = False
        if nl_query_text:
            if self.nl_embedder.is_fitted:
                q_nl_vec = self.nl_embedder.transform([nl_query_text])[0]
                # Query NL index
                nl_res = self.nl_index.query(q_nl_vec.tolist(), k=topk*2)
                for eid, score in nl_res:
                    nl_scores[eid] = score
                has_nl_query = True
                if not nl_res:
                    import logging
                    logging.getLogger(__name__).warning("NL index returned 0 results for query.")
            else:
                import logging
                logging.getLogger(__name__).warning("NL query provided but NL embedder/index not ready (is_fitted=False).")

        # D. Determine Alpha
        meta = self.db_meta
        
        used_alpha = meta.retrieval_alpha_default
        
        if not has_struct:
            used_alpha = 0.0
        elif alpha is not None:
            used_alpha = clamp_alpha(alpha)
        elif use_learned_alpha and meta.alpha_model_enabled and self.alpha_model.is_fitted:
            # Predict alpha using canonical feature extractor
            clean_feats = extract_alpha_features(prof, family="unknown", has_nl=has_nl_query)
            used_alpha = self.alpha_model.predict_alpha(clean_feats)
        
        if not has_nl_query and has_struct:
             used_alpha = 1.0
        
        # E. Fusion
        fused_scores = fuse_scores(struct_scores, nl_scores, used_alpha)
        
        # Sort candidates by fused score
        sorted_cand = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [eid for eid, _ in sorted_cand[:topk*3]] # Pool for MMR
        
        # F. Diversification (MMR)
        id_to_vec = {} 
        try:
            if hasattr(self.index, "ids") and hasattr(self.index, "vectors"):
                 id_map = {eid: i for i, eid in enumerate(self.index.ids)}
                 for cid in top_candidates:
                     idx = id_map.get(cid)
                     if idx is not None:
                         id_to_vec[cid] = self.index.vectors[idx]
        except:
             pass
             
        # MMR Select
        selected_ids = mmr_select(top_candidates, fused_scores, id_to_vec, topk * 2) # Get more for dedupe
        
        # G. Deduplicate and Format Results
        results_map = {} # dedupe_key -> result_dict
        
        for eid in selected_ids:
            rec = self.store.get_entry_record(eid)
            if not rec: continue
            
            dk = entry_dedupe_key(rec.meta, rec.hashes)
            score = fused_scores.get(eid, 0.0)
            
            # If duplicate, keep higher score (or first encountered if equal - entry_id tie-break is implicit in iteration order)
            if dk in results_map:
                if score > results_map[dk]["final_score"]:
                    pass # Replace
                else:
                    continue # Keep existing
            
            results_map[dk] = {
                "entry_id": eid,
                "problem_id": rec.meta.problem_id,
                "dataset_id": rec.meta.dataset_id,
                "family": rec.meta.family,
                "final_score": score,
                "structural_similarity": struct_scores.get(eid, 0.0),
                "nl_similarity": nl_scores.get(eid, 0.0),
                "alpha_used": used_alpha,
                "dedupe_key": dk,
                "has_nl": bool(rec.meta.nl_text)
            }
            
        # Final Sort and Slice
        final_results = sorted(results_map.values(), key=lambda x: x["final_score"], reverse=True)
        return final_results[:topk]

    def rebuild_nl_index(self):
        """Refits NL embedder on all texts and rebuilds index."""
        texts = []
        ids = []
        for em in self.store.iter_entry_metas():
            t = em.nl_text or ""
            texts.append(t)
            ids.append(em.entry_id)
            
        if not texts:
            return
        
        self.nl_embedder.fit(texts)
        self.nl_embedder.save(self.nl_embedder_path)
        
        vecs = self.nl_embedder.transform(texts)
        self.nl_index = VectorIndex(cache_dir=self.cache_dir / "nl") 
        if ids:
            self.nl_index.add(vecs.tolist(), ids)
            
        self.nl_index.save(self.nl_index_path)
        
        meta = self.db_meta
        meta.nl_index_dirty = False
        self.store.set_meta(meta)
        
    def rebuild_alpha_model(self):
        """Retrains AlphaModel."""
        profs = self.store.load_all_profiles()
        examples = []
        for em in self.store.iter_entry_metas():
            prof = profs.get(em.entry_id)
            if not prof: continue
            
            has_nl = 1 if em.nl_text else 0
            
            # Extract scalar stats for heuristics
            # Profile structure from dictionary (json)
            stats = prof.get("cnf_stats", {})
            n_vars = stats.get("n_vars", 0)
            n_clauses = stats.get("n_clauses", 0)
            family = em.family.lower()
            
            # Label heuristics
            alpha_label = 0.70
            if not em.nl_text:
                alpha_label = 1.0
            elif stats.get("n_vars", 0) >= 200 or stats.get("n_clauses", 0) >= 2000:
                alpha_label = 0.85
            elif any(x in em.family.lower() for x in ["logic", "grid", "puzzle"]):
                alpha_label = 0.55
            else:
                alpha_label = 0.70
            
            # Use canonical feature extractor
            p_obj = ConstraintProfile(**prof) if isinstance(prof, dict) else prof
            clean_feats = extract_alpha_features(p_obj, family=em.family, has_nl=bool(em.nl_text))
            
            examples.append(AlphaExample(features=clean_feats, alpha=alpha_label))
            
        if examples:
            self.alpha_model.fit(examples)
            self.alpha_model.save(self.alpha_model_path)
            
            meta = self.db_meta
            meta.alpha_model_dirty = False
            self.store.set_meta(meta)
        

    def scan_integrity(self) -> Any:
        """Scans the DB for issues."""
        from Denabase.db.integrity import IntegrityChecker
        checker = IntegrityChecker(self.store)
        return checker.scan()

    def rebuild_index(self) -> int:
        """Rebuilds all indexes from source artifacts."""
        from Denabase.db.integrity import IntegrityChecker
        # Reset vector index
        self.index = VectorIndex()
        checker = IntegrityChecker(self.store)
        count = checker.repair_indexes(self.embedder, self.index)
        self.index.save(self.index_path)
        
        meta = self.db_meta
        meta.structural_index_dirty = False
        self.store.set_meta(meta)
        
        return count

    def attach_trace(self, entry_id: str, trace: EncodingTrace) -> None:
        """Attaches a trace to an existing entry."""
        rec = self.store.get_entry_record(entry_id)
        if not rec:
            raise ValueError(f"Entry {entry_id} not found")
        
        # Save trace artifact
        rel_path = self.store.save_trace(entry_id, trace)
        
        # Update metadata
        rec.meta.trace_path = rel_path
        rec.meta.has_trace = True
        
        # Save updated record
        self.store.put_entry(rec, {})

    def get_trace(self, entry_id: str) -> Optional[EncodingTrace]:
        """Retrieves the trace for an entry."""
        return self.store.load_trace(entry_id)
