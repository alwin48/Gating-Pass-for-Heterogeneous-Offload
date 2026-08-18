; Canonical affine saxpy-like loop (GPU-safe candidate)
; for (i=0; i<N; ++i) C[i] = A[i] + B[i];  with N=1024

define void @vec_add(ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture writeonly %C) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %a.ptr = getelementptr inbounds double, ptr %A, i64 %i
  %b.ptr = getelementptr inbounds double, ptr %B, i64 %i
  %c.ptr = getelementptr inbounds double, ptr %C, i64 %i
  %a = load double, ptr %a.ptr, align 8
  %b = load double, ptr %b.ptr, align 8
  %sum = fadd double %a, %b
  store double %sum, ptr %c.ptr, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 1024
  br i1 %cond, label %exit, label %for.body

exit:
  ret void
}
