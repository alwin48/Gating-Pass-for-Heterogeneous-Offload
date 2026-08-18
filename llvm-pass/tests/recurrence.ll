; Loop-carried RAW recurrence (NOT GPU-safe)
; for (i=1; i<N; ++i) A[i] = A[i-1] + B[i];

define void @recurrence(ptr nocapture %A, ptr nocapture readonly %B) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 1, %entry ], [ %i.next, %for.body ]
  %im1 = add nsw i64 %i, -1
  %a.im1.ptr = getelementptr inbounds double, ptr %A, i64 %im1
  %b.ptr = getelementptr inbounds double, ptr %B, i64 %i
  %a.i.ptr = getelementptr inbounds double, ptr %A, i64 %i
  %a.im1 = load double, ptr %a.im1.ptr, align 8
  %b = load double, ptr %b.ptr, align 8
  %sum = fadd double %a.im1, %b
  store double %sum, ptr %a.i.ptr, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 1024
  br i1 %cond, label %exit, label %for.body

exit:
  ret void
}
