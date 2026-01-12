import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from adalflow.core.generator import BackwardEngine

class TRECTrainingPipeline(adal.AdalComponent):
    """
    Orchestrates the training loop: Forward -> Eval -> Loss (Gradient).
    """
    def __init__(self, student_task, teacher_client, teacher_model_kwargs):
        
        # 1. Metric: Exact Match 
        # Checks if the parsed string exactly matches the Ground Truth label (e.g., "LOC" == "LOC").
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # 2. Textual Loss Function
        # Converts a failure (score 0) into a textual prompt for the Teacher.
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Check if the predicted class label matches the ground truth label exactly."
        )
        
        # 3. Configure Teacher (Backward Engine)
        teacher_config = {"model_client": teacher_client, "model_kwargs": teacher_model_kwargs}
        teacher_engine = BackwardEngine(**teacher_config)
        loss_fn.set_backward_engine(backward_engine=teacher_engine)

        super().__init__(
            task=student_task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            text_optimizer_model_config=teacher_config,
            backward_engine_model_config=teacher_config
        )

    def prepare_task(self, sample):
        """Prepares input for the Student's forward pass."""
        return self.task.call, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample, y_pred):
        """Evaluates prediction and optionally logs errors."""
        # Log only errors to keep console output clean
        if y_pred.data != sample.class_name:
            print(f"❌ Q: {sample.question} | Pred: {y_pred.data} | GT: {sample.class_name}")
            
        return self.eval_fn, {"y": y_pred.data, "y_gt": sample.class_name}

    def prepare_loss(self, sample, pred):
        """Prepares input for the Teacher's backward pass (if error occurred)."""
        # Wrap ground truth as a Parameter for the loss function
        y_gt = adal.Parameter(name="y_gt", data=sample.class_name, requires_opt=False)
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}