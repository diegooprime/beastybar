#!/bin/bash
# AWS Tablebase Instance - Check Quota and Launch
# Usage: ./aws_check_and_launch.sh [check|launch|status|terminate]

set -e

INSTANCE_TYPE="c7i.48xlarge"
KEY_NAME="tablebase-key"
SECURITY_GROUP="sg-05821586abd867932"
AMI_ID="ami-0b6c6ebed2801a5cb"
QUOTA_REQUEST_ID="1c12e032b7ee4c988c176b850b68a9c90uL5knAD"

check_quota() {
    echo "Checking quota increase status..."
    STATUS=$(aws service-quotas get-requested-service-quota-change \
        --request-id "$QUOTA_REQUEST_ID" \
        --query 'RequestedQuota.Status' \
        --output text 2>/dev/null || echo "ERROR")

    echo "Status: $STATUS"

    if [ "$STATUS" = "CASE_OPENED" ] || [ "$STATUS" = "APPROVED" ]; then
        echo "✓ Quota increase approved! You can now launch the instance."
        echo "Run: ./aws_check_and_launch.sh launch"
        return 0
    elif [ "$STATUS" = "PENDING" ]; then
        echo "⏳ Still pending... Check again in 30 minutes."
        return 1
    elif [ "$STATUS" = "DENIED" ]; then
        echo "✗ Quota increase denied. You may need to contact AWS support."
        return 2
    else
        echo "? Unknown status: $STATUS"
        return 1
    fi
}

launch_instance() {
    echo "Launching $INSTANCE_TYPE spot instance..."

    RESULT=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"3.50","SpotInstanceType":"one-time"}}' \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tablebase-gen}]' \
        --query 'Instances[0].InstanceId' \
        --output text)

    echo "Instance launched: $RESULT"
    echo ""
    echo "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids "$RESULT"

    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$RESULT" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    echo ""
    echo "=========================================="
    echo "✓ Instance is running!"
    echo "Instance ID: $RESULT"
    echo "Public IP: $PUBLIC_IP"
    echo ""
    echo "Connect with:"
    echo "  ssh -i ~/.ssh/tablebase-key.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "Don't forget to terminate when done:"
    echo "  ./aws_check_and_launch.sh terminate $RESULT"
    echo "=========================================="
}

get_status() {
    echo "Finding tablebase-gen instances..."
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=tablebase-gen" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[*].Instances[*].{ID:InstanceId,State:State.Name,IP:PublicIpAddress,Type:InstanceType}' \
        --output table
}

terminate_instance() {
    if [ -z "$2" ]; then
        echo "Finding tablebase-gen instances to terminate..."
        INSTANCE_ID=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=tablebase-gen" "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text)

        if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
            echo "No running tablebase-gen instances found."
            return 1
        fi
    else
        INSTANCE_ID="$2"
    fi

    echo "Terminating instance: $INSTANCE_ID"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
        --query 'TerminatingInstances[0].{ID:InstanceId,State:CurrentState.Name}' \
        --output table

    echo "✓ Instance termination initiated. Billing will stop shortly."
}

case "${1:-check}" in
    check)
        check_quota
        ;;
    launch)
        launch_instance
        ;;
    status)
        get_status
        ;;
    terminate)
        terminate_instance "$@"
        ;;
    *)
        echo "Usage: $0 [check|launch|status|terminate [instance-id]]"
        echo ""
        echo "Commands:"
        echo "  check     - Check if quota increase is approved"
        echo "  launch    - Launch the c7i.48xlarge spot instance"
        echo "  status    - Show running tablebase instances"
        echo "  terminate - Terminate the instance (stops billing)"
        ;;
esac
