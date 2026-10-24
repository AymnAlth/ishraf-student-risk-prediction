/**
 * Every feature is bounded by the snapshot date. The label is calculated only
 * from the following horizon, preventing future attendance from entering a feature.
 */
export const STUDENT_RISK_SNAPSHOT_QUERY = `
WITH cutoff_candidates AS (
  SELECT DISTINCT
    attendance.student_id,
    session.class_id,
    session.academic_year_id,
    session.semester_id,
    session.session_date AS cutoff_date
  FROM attendance
  JOIN attendance_sessions AS session
    ON session.id = attendance.attendance_session_id
  JOIN students AS student
    ON student.id = attendance.student_id
  WHERE student.status = 'active'
),
eligible_snapshots AS (
  SELECT
    candidate.student_id,
    candidate.class_id,
    candidate.academic_year_id,
    candidate.semester_id,
    candidate.cutoff_date,
    label_metrics.future_attendance_records,
    label_metrics.future_present_count
  FROM cutoff_candidates AS candidate
  CROSS JOIN LATERAL (
    SELECT
      COUNT(*)::integer AS future_attendance_records,
      COUNT(*) FILTER (WHERE future_attendance.status = 'present')::integer AS future_present_count
    FROM attendance AS future_attendance
    JOIN attendance_sessions AS future_session
      ON future_session.id = future_attendance.attendance_session_id
    WHERE future_attendance.student_id = candidate.student_id
      AND future_session.session_date > candidate.cutoff_date
      AND future_session.session_date <= candidate.cutoff_date + $2::integer
  ) AS label_metrics
  WHERE label_metrics.future_attendance_records >= $3::integer
)
SELECT
  snapshot.student_id::text,
  snapshot.cutoff_date::text,
  class.grade_level_id::text,
  GREATEST(0, snapshot.cutoff_date - student.enrollment_date)::text AS days_since_enrollment,
  attendance_metrics.attendance_records::text AS lookback_attendance_records,
  attendance_metrics.present_count::text AS lookback_present_count,
  attendance_metrics.absent_count::text AS lookback_absent_count,
  attendance_metrics.late_count::text AS lookback_late_count,
  attendance_metrics.excused_count::text AS lookback_excused_count,
  attendance_metrics.attendance_percent::text AS lookback_attendance_percent,
  assessment_metrics.assessment_count::text AS lookback_assessment_count,
  assessment_metrics.assessment_average_percent::text AS lookback_assessment_average_percent,
  assessment_metrics.assessment_minimum_percent::text AS lookback_assessment_minimum_percent,
  homework_metrics.homework_due_count::text AS lookback_homework_due_count,
  homework_metrics.homework_completed_count::text AS lookback_homework_completed_count,
  homework_metrics.homework_late_count::text AS lookback_homework_late_count,
  homework_metrics.homework_missing_count::text AS lookback_homework_missing_count,
  behavior_metrics.negative_behavior_count::text AS lookback_negative_behavior_count,
  behavior_metrics.positive_behavior_count::text AS lookback_positive_behavior_count,
  behavior_metrics.negative_behavior_severity_sum::text AS lookback_negative_behavior_severity_sum,
  transport_metrics.assigned_at_snapshot AS transport_assigned_at_snapshot,
  transport_metrics.boarded_count::text AS lookback_transport_boarded_count,
  transport_metrics.absent_count::text AS lookback_transport_absent_count,
  snapshot.future_attendance_records::text AS label_observation_count,
  ROUND(
    100.0 * snapshot.future_present_count / NULLIF(snapshot.future_attendance_records, 0),
    2
  )::text AS label_future_attendance_percent,
  (
    100.0 * snapshot.future_present_count / NULLIF(snapshot.future_attendance_records, 0)
  ) < $4::numeric AS label_at_risk_next_14d
FROM eligible_snapshots AS snapshot
JOIN students AS student
  ON student.id = snapshot.student_id
LEFT JOIN classes AS class
  ON class.id = snapshot.class_id
CROSS JOIN LATERAL (
  SELECT
    COUNT(*)::integer AS attendance_records,
    COUNT(*) FILTER (WHERE history_attendance.status = 'present')::integer AS present_count,
    COUNT(*) FILTER (WHERE history_attendance.status = 'absent')::integer AS absent_count,
    COUNT(*) FILTER (WHERE history_attendance.status = 'late')::integer AS late_count,
    COUNT(*) FILTER (WHERE history_attendance.status = 'excused')::integer AS excused_count,
    ROUND(
      100.0 * COUNT(*) FILTER (WHERE history_attendance.status = 'present') / NULLIF(COUNT(*), 0),
      2
    ) AS attendance_percent
  FROM attendance AS history_attendance
  JOIN attendance_sessions AS history_session
    ON history_session.id = history_attendance.attendance_session_id
  WHERE history_attendance.student_id = snapshot.student_id
    AND history_session.session_date >= snapshot.cutoff_date - ($1::integer - 1)
    AND history_session.session_date <= snapshot.cutoff_date
) AS attendance_metrics
CROSS JOIN LATERAL (
  SELECT
    COUNT(*)::integer AS assessment_count,
    ROUND(
      AVG(100.0 * student_assessment.score / NULLIF(assessment.max_score, 0)),
      2
    ) AS assessment_average_percent,
    ROUND(
      MIN(100.0 * student_assessment.score / NULLIF(assessment.max_score, 0)),
      2
    ) AS assessment_minimum_percent
  FROM student_assessments AS student_assessment
  JOIN assessments AS assessment
    ON assessment.id = student_assessment.assessment_id
  WHERE student_assessment.student_id = snapshot.student_id
    AND assessment.assessment_date >= snapshot.cutoff_date - ($1::integer - 1)
    AND assessment.assessment_date <= snapshot.cutoff_date
    AND student_assessment.graded_at IS NOT NULL
    AND student_assessment.graded_at::date <= snapshot.cutoff_date
) AS assessment_metrics
CROSS JOIN LATERAL (
  SELECT
    COUNT(*)::integer AS homework_due_count,
    COUNT(*) FILTER (
      WHERE homework_submission.status IN ('submitted', 'late')
    )::integer AS homework_completed_count,
    COUNT(*) FILTER (WHERE homework_submission.status = 'late')::integer AS homework_late_count,
    COUNT(*) FILTER (
      WHERE homework_submission.id IS NULL OR homework_submission.status = 'not_submitted'
    )::integer AS homework_missing_count
  FROM homework
  LEFT JOIN homework_submissions AS homework_submission
    ON homework_submission.homework_id = homework.id
    AND homework_submission.student_id = snapshot.student_id
    AND homework_submission.updated_at::date <= snapshot.cutoff_date
  WHERE homework.class_id = snapshot.class_id
    AND homework.assigned_date >= snapshot.cutoff_date - ($1::integer - 1)
    AND homework.assigned_date <= snapshot.cutoff_date
    AND homework.due_date <= snapshot.cutoff_date
) AS homework_metrics
CROSS JOIN LATERAL (
  SELECT
    COUNT(*) FILTER (WHERE behavior_category.behavior_type = 'negative')::integer AS negative_behavior_count,
    COUNT(*) FILTER (WHERE behavior_category.behavior_type = 'positive')::integer AS positive_behavior_count,
    COALESCE(
      SUM(behavior_record.severity) FILTER (WHERE behavior_category.behavior_type = 'negative'),
      0
    )::integer AS negative_behavior_severity_sum
  FROM behavior_records AS behavior_record
  JOIN behavior_categories AS behavior_category
    ON behavior_category.id = behavior_record.behavior_category_id
  WHERE behavior_record.student_id = snapshot.student_id
    AND behavior_record.behavior_date >= snapshot.cutoff_date - ($1::integer - 1)
    AND behavior_record.behavior_date <= snapshot.cutoff_date
) AS behavior_metrics
CROSS JOIN LATERAL (
  SELECT
    EXISTS (
      SELECT 1
      FROM student_bus_assignments AS assignment
      WHERE assignment.student_id = snapshot.student_id
        AND assignment.start_date <= snapshot.cutoff_date
        AND (assignment.end_date IS NULL OR assignment.end_date >= snapshot.cutoff_date)
    ) AS assigned_at_snapshot,
    COUNT(*) FILTER (WHERE trip_event.event_type = 'boarded')::integer AS boarded_count,
    COUNT(*) FILTER (WHERE trip_event.event_type = 'absent')::integer AS absent_count
  FROM trip_student_events AS trip_event
  WHERE trip_event.student_id = snapshot.student_id
    AND trip_event.event_time::date >= snapshot.cutoff_date - ($1::integer - 1)
    AND trip_event.event_time::date <= snapshot.cutoff_date
) AS transport_metrics
ORDER BY snapshot.student_id, snapshot.cutoff_date
`;
