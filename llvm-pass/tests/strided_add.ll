; Strided vector add (stride 16) — typically unprofitable on GPU

define void @strided_add(ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture writeonly %C) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %idx = mul nuw nsw i64 %i, 16
  %a.ptr = getelementptr inbounds double, ptr %A, i64 %idx
  %b.ptr = getelementptr inbounds double, ptr %B, i64 %idx
  %c.ptr = getelementptr inbounds double, ptr %C, i64 %idx
  %a = load double, ptr %a.ptr, align 8
  %b = load double, ptr %b.ptr, align 8
  %sum = fadd double %a, %b
  store double %sum, ptr %c.ptr, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 4096
  br i1 %cond, label %exit, label %for.body

exit:
  ret void
}
